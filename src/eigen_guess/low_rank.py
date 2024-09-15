# """This module provides functionality for low-rank approximations."""
# import numpy as np
# import torch
# import torch.nn as nn


# def low_rank_approximate(mat_org: torch.tensor, rank=32):
#     """Learning a low-rank decomposition for the given matrix.

#     Args:
#         mat_org (torch.tensor): the given matrix.
#         rank (int, optional): defined rank value. Defaults to 16.
#     """
#     device = mat_org.device

#     if not device == 'cpu':
#         mat_org = mat_org.cpu()
#     u, s, vh = np.linalg.svd(mat_org.detach().numpy(), full_matrices=True)

#     s_val = np.sqrt(np.diag(s[:rank])) # half singular value
#     mat_q = torch.tensor(u[:, :rank] @ s_val)
#     mat_r = torch.tensor(s_val @ vh[:rank, :])
#     error = nn.functional.mse_loss(mat_q @ mat_r, mat_org)

#     mat_q = mat_q.to(device)
#     mat_r = mat_r.to(device)

#     output = {'mat_q': mat_q,
#               'mat_r': mat_r.t(),
#               'error': error}
#     return output


# class ModuleLowRank(object):
#     """Replace the original Linear matrix with two low-rank linear matrices.

#     Args:
#         compress_ratio (int): the pre-defined rank ratio value.
#         name_omit (list of str): the omitted name list for low-rank approximation.
#         is_approximate (bool, optional): perform low-rank approximation. Defaults to True.
#     """

#     def __init__(self,
#                  compress_ratio=3,
#                  name_omit=None,
#                  is_approximate=True):
#         """Initialize the ModuleLowRank object."""
#         if name_omit is None:
#             name_omit = []
#         super().__init__()
#         self.compress_ratio = compress_ratio
#         self.name_omit = name_omit
#         self.is_approximate = is_approximate

#     def _apply(self, name: str, module: nn.Linear):
#         """Apply nn.Sequential for replacement of the Linear module.

#         Args:
#             name (str): module name
#             module (nn.Linear): the given Linear module
#         """
#         shape = (module.in_features, module.out_features)
#         weight, bias = module.weight, module.bias

#         rank = (shape[0] * shape[1]) // (self.compress_ratio * (shape[0] + shape[1]))
#         rank = int(rank)

#         # Add two new Linear modules
#         module_l = nn.Linear(shape[0], rank, bias=False,)
#         module_r = nn.Linear(rank, shape[1], bias=bias is not None,)
#         module_l = module_l.to(weight.device) # for old pytorch version
#         module_r = module_r.to(weight.device) # for old pytorch version

#         if self.is_approximate:
#             lr_out = low_rank_approximate(weight.t(), rank)
#             weight_l, weight_r = lr_out['mat_q'], lr_out['mat_r']

#             module_l.weight.data.copy_(weight_l.t())
#             module_r.weight.data.copy_(weight_r)
#             if bias is not None:
#                 module_r.bias.data.copy_(bias)
#         else:
#             weight_l, weight_r = None, None

#         return {'weight_l': weight_l,
#                 'weight_r': weight_r,
#                 'module_rep': nn.Sequential(module_l, module_r)}

#     def __call__(self, module: nn.Module):
#         """Apply low-rank approximation to all Linear modules in the given model.

#         Args:
#             module (nn.Module): The model to apply low-rank approximation to.

#         Returns:
#             nn.Module: The modified model with low-rank approximated Linear modules.
#         """
#         copied_modules = dict(module.named_modules())
#         for name, module_sub in copied_modules.items():
#             if isinstance(module_sub, nn.Linear):
#                 if any(n in name for n in self.name_omit):
#                     continue
#                 if module_sub.out_features < 10:
#                     continue # for some head matrix, such as image-text match head

#                 base, localname = module, name
#                 while '.' in localname:
#                     prefix, localname = localname.split('.', 1)
#                     base = base.__getattr__(prefix)
#                 output = self._apply(name, module_sub)
#                 print("applying low rank on", name)

#                 setattr(base, localname, output['module_rep'])

#         return module

import numpy as np
import torch
import torch.nn as nn


def svd_decomposition(mat_org: torch.Tensor) -> tuple:
    """Perform SVD decomposition on the matrix."""
    return np.linalg.svd(mat_org.detach().numpy(), full_matrices=True)

def compute_low_rank_matrices(u, s, vh, rank: int) -> tuple:
    """Compute low-rank matrices from SVD components."""
    s_val = np.sqrt(np.diag(s[:rank]))  # half singular value
    mat_q = torch.tensor(u[:, :rank] @ s_val)
    mat_r = torch.tensor(s_val @ vh[:rank, :])
    return mat_q, mat_r

def low_rank_approximate(mat_org: torch.Tensor, rank=32) -> dict:
    """Learning a low-rank decomposition for the given matrix."""
    device = mat_org.device

    if device != 'cpu':
        mat_org = mat_org.cpu()

    u, s, vh = svd_decomposition(mat_org)
    mat_q, mat_r = compute_low_rank_matrices(u, s, vh, rank)
    error = nn.functional.mse_loss(mat_q @ mat_r, mat_org)

    mat_q = mat_q.to(device)
    mat_r = mat_r.to(device)

    return {'mat_q': mat_q, 'mat_r': mat_r.t(), 'error': error}


def create_low_rank_modules(shape: tuple, rank: int, bias: bool) -> tuple:
    """Create low-rank Linear modules."""
    module_l = nn.Linear(shape[0], rank, bias=False)
    module_r = nn.Linear(rank, shape[1], bias=bias)
    return module_l, module_r

def apply_low_rank_approximation(module: nn.Linear, compress_ratio: int, is_approximate: bool) -> dict:
    """Apply low-rank approximation to a Linear module.

    Parameters:
    - module (nn.Linear): The Linear module to apply the approximation to.
    - compress_ratio (int): The ratio by which to compress the matrix.
    - is_approximate (bool): Flag to indicate whether to use approximation or not.

    Returns:
    - dict: A dictionary containing:
      - 'weight_l': The low-rank approximation of the left matrix (if approximation is applied).
      - 'weight_r': The low-rank approximation of the right matrix (if approximation is applied).
      - 'module_rep': A new nn.Sequential module containing the low-rank Linear layers.
    """
    # Get the shape of the weight matrix
    shape = (module.in_features, module.out_features)

    # Extract the weight and bias from the module
    weight, bias = module.weight, module.bias

    # Calculate the rank for the low-rank approximation.
    # The rank is determined based on the dimensions of the weight matrix and the compression ratio.
    # The formula is derived from the desired compression ratio and the matrix dimensions.
    rank = (shape[0] * shape[1]) // (compress_ratio * (shape[0] + shape[1]))
    rank = int(rank)

    module_l, module_r = create_low_rank_modules(shape, rank, bias is not None)

    if is_approximate:
        lr_out = low_rank_approximate(weight.t(), rank)
        weight_l, weight_r = lr_out['mat_q'], lr_out['mat_r']

        module_l.weight.data.copy_(weight_l.t())
        module_r.weight.data.copy_(weight_r)
        if bias is not None:
            module_r.bias.data.copy_(bias)
    else:
        weight_l, weight_r = None, None

    return {'weight_l': weight_l, 'weight_r': weight_r, 'module_rep': nn.Sequential(module_l, module_r)}


class ModuleLowRank:
    """Replace the original Linear matrix with two low-rank linear matrices."""

    def __init__(self, compress_ratio=3, name_omit=None, is_approximate=True):
        """Initialize the ModuleLowRank object.

        Parameters:
        compress_ratio (int): The compression ratio used for low-rank approximation. Default is 3.
        name_omit (list of str): A list of names to omit during compression. Default is None.
        is_approximate (bool): Flag indicating whether the approximation is approximate or exact. Default is True.
        """
        if name_omit is None:
            name_omit = []
        self.compress_ratio = compress_ratio
        self.name_omit = name_omit
        self.is_approximate = is_approximate

    def _apply(self, name: str, module: nn.Linear) -> dict:
        """Apply nn.Sequential for replacement of the Linear module."""
        return apply_low_rank_approximation(module, self.compress_ratio, self.is_approximate)

    def __call__(self, model: nn.Module) -> nn.Module:
        """Apply low-rank approximation to all Linear modules in the model."""
        copied_modules = dict(model.named_modules())
        for name, module_sub in copied_modules.items():
            if isinstance(module_sub, nn.Linear):
                if any(n in name for n in self.name_omit) or module_sub.out_features < 10:
                    continue

                base, localname = model, name
                while '.' in localname:
                    prefix, localname = localname.split('.', 1)
                    base = getattr(base, prefix)
                output = self._apply(name, module_sub)
                print("applying low rank on", name)
                setattr(base, localname, output['module_rep'])

        return model

