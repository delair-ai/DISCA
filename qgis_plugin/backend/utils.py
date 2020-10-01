"""
Some util functions used in the backend.
"""
import itertools
import os
import warnings

import buzzard as buzz
import cv2 as cv
import numpy as np
import shapely
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)


def freeze_bn(module):
    for name, m in module.named_children():
        if len(list(m.children())) > 0:
            freeze_bn(m)
        if "bn" in name: 
            m.weight.requires_grad = False 
            m.bias.requires_grad = False 

def polygonize(input_file, output_file, proj):
    """
    Polygonise a raster file
    Parameters
    ----------
    input_file: str Path to input raster
    output_file: str Path to output vector
    proj: str Projection
    """
    with buzz.Dataset(sr_work=proj, sr_fallback='WGS84').close as ds:
        ds.open_raster("raster", input_file)
        if os.path.isfile(output_file):
            os.remove(output_file)
        fields = [{"name": "class", "type": np.int32}]
        ds.create_vector("vector", output_file, 'polygon', driver="geojson", fields=fields)
        fp = ds["raster"].fp
        mask = ds["raster"].get_data()
        for class_idx in np.unique(mask):
            if class_idx != 0:
                polygons = fp.find_polygons(mask == class_idx)
                if not polygons:
                    continue
                for poly in polygons:
                    ds["vector"].insert_data(poly, {"class": class_idx})


def vec_to_list(input_vec, n_classes, fp):
    """
    Rasterize a vector file.
    Assume that geometries have a field `class`.
    Return a list of one hot encoding rasters for each class.
    input_ortho is used to delimit the boundaries of the raster.
    Only accept shapefiles and geojson as input for the vector.
    """
    with buzz.Dataset(allow_none_geometry=True).close as ds:
        ext = input_vec.split('.')[1]
        if ext == "geojson":
            driver = "geojson"
        elif ext == "shp":
            driver = 'ESRI Shapefile'
        else:
            raise Exception("Wrong kind of input vector. Expect either geojson or shapefile.")
        ds.open_vector("poly", input_vec, driver=driver)
        rasters = [np.zeros(fp.shape, dtype=np.uint8) for i in range(n_classes)]
        #TODO iter_data warning
        for geometry, class_idx in ds.poly.iter_data('class'):
            if isinstance(geometry, shapely.geometry.point.Point):
                burned_point = np.asarray(fp.spatial_to_raster(np.asarray(geometry.xy).transpose((1, 0)))[0])
                if 0 < burned_point[1] < rasters[0].shape[0] and 0 < burned_point[0] < rasters[0].shape[1]:
                    rasters[class_idx][burned_point[1], burned_point[0]] = 1
            elif isinstance(geometry, shapely.geometry.polygon.Polygon):
                burned_polygon = fp.burn_polygons(geometry)
                rasters[class_idx][burned_polygon] = 1
        return rasters


def encode_points(rasters):
    """
    encode the points using either binary encoding or distance transform
    """
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    outputs = [cv.dilate(raster, kernel) for  raster in rasters]
    outputs = [cv.distanceTransform(raster.astype(np.uint8), cv.DIST_L2, 3) for raster in outputs]
    output = [raster[np.newaxis] for raster in outputs]
    return output


def make_batches(n, iterable):
    """ Browse an iterator by chunks of size n, yield n footprints"""
    while True:
        chunk = tuple(itertools.islice(iterable, n))
        if not chunk:
            return
        yield chunk


def from_coord_to_patch(ortho, fps, big_fp):
    """Returns an n batch of the input ortho using fps (footprints)."""
    image_patches = []
    for fp in fps:
        partial_patch = ortho[:, fp.slice_in(big_fp)[0], fp.slice_in(big_fp)[1]]
        image_patches.append(partial_patch)
    image_patches = torch.stack(image_patches, dim=0)
    return image_patches


def print_warning(msg):
    print('\033[91m' + str(msg) + '\033[0m')


def is_intersect(fp, geom):
    """Return True if the twoo geometries (or footprint) are intersected. Also work with Points contrary to fp.share_area"""
    try:
        fp.intersection(geom)
        out = True 
    except:
        out = False
    return out
