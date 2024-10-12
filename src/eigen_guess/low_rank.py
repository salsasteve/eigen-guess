"""This module provides functionality for low-rank approximations."""

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

    if device != "cpu":
        mat_org = mat_org.cpu()

    u, s, vh = svd_decomposition(mat_org)
    mat_q, mat_r = compute_low_rank_matrices(u, s, vh, rank)
    error = nn.functional.mse_loss(mat_q @ mat_r, mat_org)

    mat_q = mat_q.to(device)
    mat_r = mat_r.to(device)

    return {"mat_q": mat_q, "mat_r": mat_r.t(), "error": error}


def create_low_rank_modules(shape: tuple, rank: int, bias: bool) -> tuple:
    """Create low-rank Linear modules."""
    module_l = nn.Linear(shape[0], rank, bias=False)
    module_r = nn.Linear(rank, shape[1], bias=bias)
    return module_l, module_r


def apply_low_rank_approximation(
    module: nn.Linear, compress_ratio: int, is_approximate: bool
) -> dict:
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
        weight_l, weight_r = lr_out["mat_q"], lr_out["mat_r"]

        module_l.weight.data.copy_(weight_l.t())
        module_r.weight.data.copy_(weight_r)
        if bias is not None:
            module_r.bias.data.copy_(bias)
    else:
        weight_l, weight_r = None, None

    return {
        "weight_l": weight_l,
        "weight_r": weight_r,
        "module_rep": nn.Sequential(module_l, module_r),
    }


class ModuleLowRank:
    """Replace the original Linear matrix with two low-rank linear matrices."""

    def __init__(
        self,
        compress_ratio=3,
        name_omit=None,
        is_approximate=True,
        num_layers=-1,
    ):
        """Initialize the ModuleLowRank object.

        Parameters:
        - compress_ratio (int): The compression ratio used for low-rank approximation. Default is 3.
        - name_omit (list of str): A list of names to omit during compression. Default is None.
        - is_approximate (bool): Flag indicating whether the approximation is approximate or exact. Default is True.
        - num_layers (int): Number of layers to apply low-rank approximation. Default is -1 (all layers).
        """
        if name_omit is None:
            name_omit = []
        self.compress_ratio = compress_ratio
        self.name_omit = name_omit
        self.is_approximate = is_approximate
        self.num_layers = (
            num_layers  # New parameter to specify the number of layers
        )

    def _apply(self, name: str, module: nn.Linear) -> dict:
        """Apply nn.Sequential for replacement of the Linear module."""
        return apply_low_rank_approximation(
            module, self.compress_ratio, self.is_approximate
        )

    def __call__(self, model: nn.Module) -> nn.Module:
        """Apply low-rank approximation to specified number of Linear modules in the model."""
        # Create a list of all eligible Linear modules
        linear_modules = []
        for name, module_sub in model.named_modules():
            if isinstance(module_sub, nn.Linear):
                if (
                    any(n in name for n in self.name_omit)
                    or module_sub.out_features < 10
                ):
                    continue
                linear_modules.append((name, module_sub))

        # Determine how many layers to compress
        num_layers_to_compress = (
            self.num_layers if self.num_layers != -1 else len(linear_modules)
        )
        num_layers_to_compress = min(
            num_layers_to_compress, len(linear_modules)
        )

        # Apply low-rank approximation to the specified number of layers
        for idx, (name, module_sub) in enumerate(linear_modules):
            if idx >= num_layers_to_compress:
                break  # Stop after compressing the specified number of layers

            # Navigate to the parent module to replace the submodule
            base, localname = model, name
            if "." in localname:
                names = localname.split(".")
                for n in names[:-1]:
                    base = getattr(base, n)
                localname = names[-1]

            output = self._apply(name, module_sub)
            print(f"Applying low-rank approximation on layer: {name}")
            setattr(base, localname, output["module_rep"])

        return model
