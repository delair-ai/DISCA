
import os
import warnings
from multiprocessing.connection import Client, Listener
from time import time

import buzzard as buzz
import GPUtil
import numpy as np
import rasterio as rio
import torch
import yaml
from shapely.geometry import box
from torch import jit
from tqdm import tqdm

from .learning_net import LearningNet
from .ssh_connexion import SshConnexion
from .utils import (encode_points, from_coord_to_patch, is_intersect,
                    make_batches, polygonize, print_warning, vec_to_list)

warnings.simplefilter(action='ignore', category=UserWarning)

class Daemon:
    """Connect a daemon to a client and execute computer vision tasks in it.
    """

    def __init__(
        self,
        config_file="backend/config.yml",
        ssh=False,
        cache=True,
        connexion_file="connexion_setup.yml",
        cpu=False,
    ):
        with open(config_file, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.connexion_file = connexion_file
        self.ssh_server = SshConnexion(connexion_file) if ssh else None
        device = self._set_device(cpu)
        print(f"\u001b[36mDevice: {device}\033[0m")
        self.device = torch.device(device)
        self.cache = {} if cache else None
        self.original_fp = None
        self.n_classes = None
        self.tiled_fp = None
        self.idxs_overlap = None
        self.first_start = True
        self.annotations = []
        self.continual_learn = None
        self.cl_opts = {}
        # Continual Learning: implie that the new pred is computed on all footprints;
        #  not just the ones surrounding new annot points.

    def _save_output(self, data, output, nodata_mask, annot_tiles, pred_fp):
        """Save outputs"""
        if self.first_start:
            fp = self.original_fp
        else:
            fps = [self.tiled_fp[i] for i in annot_tiles]
            x = (min([i.gt[0] for i in fps]), max([i.gt[0] for i in fps])+ fps[0].pxsize[0]*fps[0].shape[0])
            y = (min([i.gt[3] for i in fps]) - fps[0].pxsize[1]*fps[0].shape[1], max([i.gt[3] for i in fps]))
            gt = fps[0].gt
            gt[0] = x[0]
            gt[3] = y[1]
            rsize = (np.round((x[1]-x[0])/fps[0].pxsize[0]), np.round((y[1]-y[0])/fps[0].pxsize[1]))
            fp = buzz.Footprint(gt=gt, rsize=rsize)
        input_file = data["input_files"][0]
        output_file = data["output_file"]
        if self.ssh_server:
            input_file = self.ssh_server.tmp_file(input_file)
            output_file = self.ssh_server.tmp_file(output_file)
        with buzz.Dataset().close as ds:
            proj = ds.aopen_raster(input_file).proj4_virtual
            channels_schema = {'nodata': self.n_classes}
            if os.path.isfile(output_file):
                ds.open_raster("output", output_file, mode='w')
                if not ds.output.fp.poly.contains(fp.poly):
                    ds.output.close()
                    os.remove(output_file)
                    ds.create_raster("output", output_file, self.original_fp, dtype=np.uint8, channel_count=1, sr=proj, channels_schema=channels_schema)
            else:
                ds.create_raster("output", output_file, self.original_fp, dtype=np.uint8, channel_count=1, sr=proj, channels_schema=channels_schema)

            nodata_mask = nodata_mask[fp.slice_in(self.original_fp)]
            # output[nodata_mask] = self.n_classes
            del fp
            if self.first_start:
                o = ds.output.get_data(fp=self.original_fp, channels=0)
                output[output==-1] = o[output==-1]
                ds.output.set_data(output.astype(np.uint8), self.original_fp, channels=0)
            else:
                for fp in fps:   
                    o = ds.output.get_data(fp=fp, channels=0)
                    # output[output==-1] = o[output==-1]
                    ds.output.set_data(output.astype(np.uint8)[fp.slice_in(pred_fp)], fp, channels=0)
            if not self.cl_opts["reg_L1"]:
                self.cache["initial_pred"] = torch.unsqueeze(torch.from_numpy(ds.output.get_data(self.original_fp, channels=0)), dim=0).to(self.device)
            ds.output.close()
        if self.ssh_server:
            self.ssh_server.put(data["output_file"])
        if data["polygonize"]:
            poylgon_file = data["polygonize"] if not self.ssh_server else self.ssh_server.tmp_file(data["polygonize"])
            polygonize(output_file, poylgon_file, proj)
            if self.ssh_server:
                self.ssh_server.put(data["polygonize"])

    @staticmethod
    def _set_device(cpu=False, threshold=1000):
        """Set gpu device when cuda is activated based of free available memory. 
        ---------
        Parameters:
            threshold: int Minimal amount of free memory (Mo) to select this device"""
        if cpu or not torch.cuda.is_available():
            return "cpu"
        for d, i in enumerate(GPUtil.getGPUs()):
            if i.memoryFree > threshold: 
                device = d
                break
            elif d + 1 == len(GPUtil.getGPUs()):
                return "cpu"
        return f"cuda:{device}"

    def _prepare_inputs(self, data, task):
        """
        Open rasters with buzzard and normalize them if it's uint (assuming it's RGB which needs to be normalized).
        Put data in cache if type(self.cache)==dict.
        Returned footprint: 
            -   Original if default option and no annots
            -   BB englobing annots if there are annots
            -   Manual bb if user has specified one
        """
        assert self.n_classes
        input_files = data["input_files"]
        if not data["interact_use_annots"] or data["reset"] or data["reset_network"]:
            # reset annotation counter to make a full new prediction
            self.cache["initial_pred"] = None
            if isinstance(self.tiled_fp, list):
                self.tiled_fp = [[i[0], 0] for i in self.tiled_fp]
        if isinstance(self.cache, dict) and input_files != self.cache.get("input_files"):
            if self.ssh_server:
                for i, file in enumerate(input_files):
                    input_files[i] = self.ssh_server.get(file, cache=True)
            file = buzz.open_raster(input_files[0])
            fp = file.fp
            inputs = [buzz.open_raster(i).get_data(fp=fp, channels=[0, 1, 2]).transpose((2, 0, 1)) for i in input_files]
            nodata_mask = inputs[0][0] == buzz.open_raster(input_files[0]).nodata
            for i, raster in enumerate(inputs):
                if raster.dtype == np.uint8:
                    inputs[i] = np.asarray(raster / 255, dtype=np.float32)
            self.tiled_fp = None  # new input => reset the tiled footprint
            if isinstance(self.cache, dict):
                self.cache["input_files"] = input_files
                self.cache["inputs"] = inputs
                self.cache["fp"] = fp
                self.cache["nodata_mask"] = nodata_mask
                self.cache["initial_pred"] = None
        else:
            inputs, fp, nodata_mask = self.cache["inputs"], self.cache["fp"], self.cache["nodata_mask"]
        self.original_fp = fp
        raw_annots = None
        if data["interactive"]:
            if data["annot_layer"] is not None and data["interact_use_annots"]:
                file = data["annot_layer"]
                if self.ssh_server and file.endswith(".shp"):
                    for ext in ["shx", "cpg", "dbf", "prj", "qpj"]:
                        _ = self.ssh_server.get(file.replace('.shp', f'.{ext}'))
                annot_file = self.ssh_server.get(file) if self.ssh_server else file
                raw_annots = vec_to_list(annot_file, self.n_classes, fp)
                annots = encode_points(raw_annots)
            else:
                shape = fp.shape
                annots = [np.zeros((1, *shape)) for i in range(self.n_classes)]
                raw_annots = np.stack(annots, axis=0)
            if np.sum(raw_annots) == 0 and self.continual_learn and self.cache["initial_pred"] is not None:
                print("\u001b[33m\tLack of user annotations: Nothing to learn now.\033[0m")
                self.continual_learn = False
        else:
            annots = []
        inputs_annots = inputs + annots
        inputs_annots = [torch.from_numpy(i).to(self.device, torch.float) for i in inputs_annots]
        raw_annots = np.stack(raw_annots, axis=0)
        raw_annots = torch.from_numpy(raw_annots).to(self.device)
        return  inputs_annots, nodata_mask, raw_annots

    def segmentation(self, data):
        """
        Perform semantic segmentation
        Parameters
        ----------
        data: dict contain:
            input_files: str path to input layers
            neural_network: str path to neural network
            output_file: str path where to save the output

        Returns
        -------
        True if segmentation has been correctly perfomed. Raise an exception otherwise.
        """
        cfg = self.cfg["segmentation"]
        neural_net = data["neural_network"]
        step = 0
        print(f"\tStep {step}: Load inputs.")
        step += 1
        if self.ssh_server:
            neural_net = self.ssh_server.get(neural_net, cache=True)

        if isinstance(self.cache, dict) and self.cache.get("net_file") == neural_net and not data["reset_network"]:
            net = self.cache["net"]
            self.cache["initial_params"], _ = net.flat_params()
        else:
            net = LearningNet(neural_net, self.device)
            if isinstance(self.cache, dict):
                self.cache["net"] = net
                self.cache["net_file"] = neural_net
                self.cache["initial_pred"] = None  # used for continual learning only

        self.n_classes = net.find_n_classes()

        inputs, nodata_mask, raw_annots = self._prepare_inputs(data, "segmentation")
        if self.tiled_fp is None:
            result = self._tile_fp(cfg)
            if isinstance(result, str):
                return result
        # find the modified footprints.
        annot_file = self.ssh_server.get(data["annot_layer"]) if self.ssh_server else data["annot_layer"]
        vec_annots = buzz.open_vector(annot_file)
        annots_points = list(vec_annots.iter_data('class'))
        annot_tiles = []
        # would be possible to only save the length of the annot points and then only take the ones after that.
        # However, currently not possible if some annots points are deleted ... 
        new_points = [i for i in annots_points if i not in self.annotations]
        self.annotations = annots_points
        points = new_points if self.cl_opts["only_new_points"] else annots_points
        for i, fp in zip(self.idxs_overlap.keys(), self.tiled_fp):
            for geometry, _ in points:
                if is_intersect(fp, geometry):
                    annot_tiles.append(i)
        annot_tiles = np.unique(annot_tiles)
        pred_tiles = []
        for i in annot_tiles:
            pred_tiles.extend(self.idxs_overlap[i])
        pred_tiles = np.unique(pred_tiles)
        if pred_tiles.size == 0 and self.cache["initial_pred"] is not None:
            return "No new annotations, no inference needed."
        # check the inputs
        res_check = net.check_inputs_and_net(inputs)
        if res_check is not None:
            assert isinstance(res_check, str)
            print_warning(res_check)
            return res_check
        if self.cache["initial_pred"] is not None and self.continual_learn:
            print(f"\tStep {step}: Learning")
            step += 1
            self.continual_learn = False
            for iteration in tqdm(range(self.cl_opts["steps"]), total=self.cl_opts["steps"]):
                # update using only the patches with clicks
                if iteration > 0 and not self.continual_learn:
                    print("\u001b[33m\tLack of new user annotations: Nothing to learn now.\033[0m")
                    break
                self.continual_learn = True
                fps = [self.tiled_fp[i] for i in annot_tiles]
                for i, batch_fps in enumerate(make_batches(1, iter(fps))):
                    raw_annots_patches = from_coord_to_patch(raw_annots, batch_fps, self.original_fp)
                    raw_annots_patches = raw_annots_patches.to(torch.uint8)
                    inputs_patches = [from_coord_to_patch(x, batch_fps, self.original_fp) for x in inputs]
                    inputs_patches[1:] = [torch.zeros_like(i) for i in inputs_patches[1:]]

                    pred_patches = net.predict(torch.cat(inputs_patches, dim=1))
                    if not self.cl_opts["reg_L1"]:
                        initial_pred_patches = from_coord_to_patch(self.cache["initial_pred"].to(self.device), batch_fps, self.original_fp)
                        initial_pred_patches = initial_pred_patches[:, 0].long()
                    else:
                        initial_pred_patches = from_coord_to_patch(self.cache["initial_pred"], batch_fps, self.original_fp).float()
                    net.update(pred_patches, raw_annots_patches, initial_pred_patches, nodata=self.n_classes, opts=self.cl_opts)
                    updated_params, _ = net.flat_params()
            if self.continual_learn:
                print(f"\t\tDistance between initial parameters and updated parameters: {torch.dist(self.cache['initial_params'], updated_params)}")

        if self.cache["initial_pred"] is None and self.continual_learn:
            print("\u001b[33m\tSome features are computed; I'm learning next time.\033[0m")
        print(f"\tStep {step}: Inference")
        step += 1
        with torch.no_grad():
            self.first_start = self.cache["initial_pred"] is None or self.continual_learn
            if self.first_start:
                fps = self.tiled_fp
                f = self.original_fp
            else:
                fps = [self.tiled_fp[i] for i in pred_tiles]
                x = (min([i.gt[0] for i in fps]), max([i.gt[0] for i in fps])+ fps[0].pxsize[0]*fps[0].shape[0])
                y = (min([i.gt[3] for i in fps]) - fps[0].pxsize[1]*fps[0].shape[1], max([i.gt[3] for i in fps]))
                gt = fps[0].gt
                gt[0] = x[0]
                gt[3] = y[1]
                rsize = (np.round((x[1]-x[0])/fps[0].pxsize[0]),np.round((y[1]-y[0])/fps[0].pxsize[1]))
                f = buzz.Footprint(gt=gt, rsize=rsize)

            pred = torch.zeros((self.n_classes, *f.shape), device=self.device, dtype=torch.double)
            # Slides a window across the image
            if self.cl_opts["reg_L1"]:
                counter = torch.zeros((*f.shape), device=self.device, dtype=torch.uint8)
            for i, batch_fps in enumerate(make_batches(1, iter(fps))):
                inputs_patches = [from_coord_to_patch(x, batch_fps, self.original_fp) for x in inputs]
                # Inference
                outs = net.predict(torch.cat(inputs_patches, dim=1)) 
                outs = outs.data.to(torch.double) 
                for out, sub_fp in zip(outs, batch_fps):
                    small_sub_slice = sub_fp.slice_in(f, clip=True)
                    big_sub_slice = f.slice_in(sub_fp, clip=True)
                    pred[:, small_sub_slice[0], small_sub_slice[1]] += out[:, big_sub_slice[0], big_sub_slice[1]]
                    if self.cl_opts["reg_L1"]: 
                        counter[small_sub_slice[0], small_sub_slice[1]] += 1

        if self.cl_opts["reg_L1"] and np.all(f.shape == self.original_fp.shape):
            self.cache["initial_pred"] = (pred / counter)
        mask = torch.argmax(pred, dim=0).cpu().numpy()
        mask[pred.cpu().numpy()[0] == 0] = -1
        print(f"\tStep {step}: Save outputs")
        step += 1
        self._save_output(data, mask, nodata_mask, annot_tiles, f)
        return mask

    def save_network(self, filename):
        if self.ssh_server:
            #TODO
            print_warning('\nSaving network state not implemented yet with a ssh server.')
            return False
        try:
            self.cache["net"].net.save(filename)
            print("\033[92mNeural network saved at: \033[0m", f"\n{filename}")
            return True
        except Exception as err:
            print_warning(f'Unable to save network: {err}')
            return False

    def run(self):
        """
        Enable the server and perform some function on the inputs from the client.
        """
        print("\033[94mServer for Qgis backend running here...\033[0m") 
        if isinstance(self.cache, dict):
            print("\u001b[33mInputs and neural network will be cached in RAM.",
                  "Faster but can be troublesome with large inputs.\033[0m")
        with open(self.connexion_file, 'r') as f:
            d = yaml.safe_load(f)
            if self.ssh_server:
                address_server = tuple(d['ssh']['address_server'])
                address_client = tuple(d['ssh']['address_client'])
            else:
                address_server = tuple(d['local']['address_server'])
                address_client = tuple(d['local']['address_client'])
        while True:
            listener = Listener(address_server, authkey=b'Who you gonna call?')
            try:
                conn = listener.accept()  # ends once it has found a client
            except KeyboardInterrupt:
                print_warning("Keyboard interrupt.")
                exit()
            task, data = conn.recv()  # receive data
            conn.close()
            listener.close()
            if task == "segmentation":
                func = self.segmentation
            elif task == "save_network":
                result = self.save_network(data["refined_net_file"])
                conn_client = Client(address_client, authkey=b'ghostbusters')
                conn_client.send(result)
                conn_client.close()
                print("\u001b[35;1m\nReady for another task \U0001F60E \n\033[0m")
                continue
            else:
               return f"Task {task} not implemented."
            try:
                print(f"\u001b[36mRunning a {task.lower()} task...\033[0m")
                self.continual_learn = data['CL']
                self.cl_opts = data["cl_options"]
                tic = time()
                result = func(data)  # perform all the real work.
                toc = time()
                time_ = np.round(toc - tic, 1)
                if not isinstance(self.cache, dict) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if isinstance(result, str):
                    new_msg = result
                else:
                    new_msg = self.n_classes
                    print(f"\033[92mDone in {time_} seconds.\033[0m")
            except KeyboardInterrupt:
                new_msg = "Task interrupted by the user."
                print_warning('\n' + new_msg)
            except Exception as error:
                new_msg = error
                print_warning(error)
            finally:
                conn_client = Client(address_client, authkey=b'ghostbusters')
                conn_client.send(new_msg)
                conn_client.close()
                if isinstance(new_msg, Exception):
                    raise(new_msg)
                else:
                    print("\u001b[35;1m\nReady for another task \U0001F60E \n\033[0m")

    def _tile_fp(self, cfg):
        """compute tiled footprint and the indexes of the overlapped tiles for each tile"""
        mode = "overlap"
        try:
            tiled_fp_ = self.original_fp.tile(cfg["window_size"], cfg["stride"], cfg["stride"], mode)
            shp = tiled_fp_.shape
            tiled_fp = []
            idxs_overlap = {i: [] for i in range(np.prod(shp))}
            for x in range(shp[0]):
                for y in range(shp[1]):
                    if cfg["stride"] == 0:
                        neighbours = [(x, y)]
                    else:
                        neighbours = [(x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1), (x-1, y), (x, y-1), (x+1, y), (x, y+1), (x, y)]
                        neighbours = [i for i in neighbours if i[0] >= 0 and i[0] < shp[0] and i[1] >= 0 and i[1] < shp[1]]
                    out = [i[0] + i[1] * shp[0] for i in neighbours]
                    idxs_overlap[out[-1]].extend(out)
                    tiled_fp.append(tiled_fp_[x, y])
            tiled_fp = np.array(tiled_fp)
            self.tiled_fp, self.idxs_overlap = tiled_fp, idxs_overlap
        except ValueError:
            msg = f"Cannot segment an image with a shape smaller than {cfg['window_size']}. Input shape: {self.original_fp.rsize}"
            print_warning(msg)
            return msg
