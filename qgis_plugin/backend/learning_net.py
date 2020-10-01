import numpy as np
import torch
from torch import jit, nn, optim

IGNORE_INDEX = -1  # index ignored in the cross entropy loss (e.g. pixels which have not been clicked)


class LearningNet():
    def __init__(self, filename, device):
        self.device = device
        self.net = jit.load(filename, map_location=device)
        self.optimizer = optim.SGD(self.net.parameters(), 5*10**(-6))
        self.omega = None

    def predict(self, inputs):
        return self.net(inputs)

    def update(self, pred, raw_annots, initial_pred, nodata, opts):
        ce = not opts["reg_L1"]
        self.optimizer.zero_grad()
        sparse_target = torch.full([*pred.shape[-2:]], IGNORE_INDEX, dtype=torch.long, device=self.device)

        for i in range(raw_annots.shape[1]):
            sparse_target[raw_annots[0, i]] = i
        sparse_target = torch.unsqueeze(sparse_target, dim=0)  # add batch dimension
        loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(pred, sparse_target)
        reg = self._compute_regs(pred, initial_pred, nodata, opts)
        loss = loss + reg
        loss.backward()
        self.optimizer.step()

    def _compute_regs(self, pred, initial_pred, nodata, opts):
        reg1_fn = nn.L1Loss(reduction="none") if opts["reg_L1"] else  nn.CrossEntropyLoss(ignore_index=nodata, reduction="none")
        reg1 = opts["weight_reg"] * reg1_fn(pred, initial_pred)
        reg = torch.mean(reg1)
        return reg

    def find_n_classes(self):
        """Return the number a classes that a jit network has been trained to predict"""
        # get to the last element of the generator
        *_, p = self.net.parameters()
        n_classes = p.shape[0] if len(p.shape) == 1 else p.shape[1]
        return n_classes

    def check_inputs_and_net(self, inputs):
        """check if the inputs channels match the network input channels"""
        sum_channs = np.sum([i.shape[0] for i in inputs])
        channels_net = self.net.parameters().__next__().shape[1]
        output = None
        if sum_channs != channels_net:
            output = f"{sum_channs} input channels while the chosen net expects {channels_net} channels."
        return output

    def flat_params(self, grad=False):
        """Flatten parameters of a torch model
        Args:
            grad (bool, optional): Set to true to return the flatten gradients. Defaults to False.
        """
        params = list(self.net.parameters())
        gradients = [p.grad.view(-1) if p.grad is not None else None for p in params] if grad else None
        params = torch.cat([p.view(-1) for p in params])
        return (params, gradients)
