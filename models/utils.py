import numpy as np
import torch
from beartype import beartype
from torch import nn
from torch.nn import functional as F


class FastFoodUpsample(nn.Module):
    def __init__(self, low_dim: int, param_dims: list[tuple[int, ...]]):
        super().__init__()
        self.low_dim = low_dim
        self.param_dims = param_dims
        self.param_block_dims = [np.prod(dim) for dim in param_dims]
        self.total_dim = sum(self.param_block_dims)

        fastfood_vars = make_fastfood_vars(self.total_dim)
        for name, var in fastfood_vars.items():
            if isinstance(var, torch.Tensor):
                self.register_buffer("fastfood_" + name, var)
        self.fastfood_LL = fastfood_vars["LL"]

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        assert z.ndim == 2
        fastfood_var_dict = {
            "BB": self.fastfood_BB,
            "Pi": self.fastfood_Pi,
            "GG": self.fastfood_GG,
            "divisor": self.fastfood_divisor,
            "LL": self.fastfood_LL,
        }
        delta = fastfood_torched_batched(z, self.total_dim, fastfood_var_dict)
        delta = delta.split(self.param_block_dims, dim=1)
        delta = [d.view(-1, *dim) for d, dim in zip(delta, self.param_dims)]
        return delta


def fast_walsh_hadamard_torched_batched(
    x: torch.Tensor, axis: int = 0, normalize: bool = False
) -> torch.Tensor:
    """
    Performs fast Walsh Hadamard transform

    Args:
        x (torch.Tensor): Input matrix.
        axis (int, optional): Axis along which to perform the transform. Defaults to 0.
        normalize (bool, optional): Whether to normalize the output. Defaults to False.

    Returns:
        torch.Tensor: The output matrix of the transform.
    """
    orig_shape = x.size()[1:]
    assert axis >= 0 and axis < len(
        orig_shape
    ), "For a vector of shape %s, axis must be in [0, %d] but it is %d" % (
        orig_shape,
        len(orig_shape) - 1,
        axis,
    )
    h_dim = orig_shape[axis]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2**h_dim_exp, (
        "hadamard can only be computed over axis with size that is a "
        f"power of two, but chosen axis {axis} has size {h_dim}"
    )

    working_shape_pre = [int(np.prod(orig_shape[:axis]))]  # prod of empty array is 1 :)
    working_shape_post = [int(np.prod(orig_shape[axis + 1 :]))]  # prod of empty array is 1 :)
    working_shape_mid = [2] * h_dim_exp
    working_shape = working_shape_pre + working_shape_mid + working_shape_post

    ret = x.view(-1, *working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 2
        arrays = torch.chunk(ret, 2, dim=dim)
        assert len(arrays) == 2
        ret = torch.cat((arrays[0] + arrays[1], arrays[0] - arrays[1]), axis=dim)

    if normalize:
        ret = ret / torch.sqrt(float(h_dim))

    ret = ret.view(-1, *orig_shape)

    return ret


def make_fastfood_vars(dim: int) -> dict[str, torch.Tensor]:
    """
    Creates variables for Fastfood transform
    (an efficient random projection, see: https://arxiv.org/abs/1804.08838)

    Args:
        dim (int): Projection size of the output matrix.

    Returns:
        dict[str, torch.Tensor]: A dictionary of variables for the transform.
    """
    ll = int(np.ceil(np.log(dim) / np.log(2)))
    LL = 2**ll

    # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
    BB = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB = (BB * 2 - 1).type(torch.FloatTensor)
    BB.requires_grad = False

    # Random permutation matrix
    Pi = torch.LongTensor(np.random.permutation(LL))
    Pi.requires_grad = False

    # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
    GG = torch.FloatTensor(
        LL,
    ).normal_()
    GG.requires_grad = False

    divisor = torch.sqrt(LL * torch.sum(torch.pow(GG, 2)))

    return {"BB": BB, "Pi": Pi, "GG": GG, "divisor": divisor, "LL": LL}


def fastfood_torched_batched(
    x: torch.Tensor,
    dim: int,
    param_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Fastfood transform

    Args:
        x (torch.Tensor): Matrix that we want to project to 'dim' dimensions.
        dim int: Dimensionality of the matrix we want to project to.
        param_dict (dict[str, torch.Tensor]): Dictionary of Fastfood
            transform variables.

    Returns:
        torch.Tensor: The Fastfood transform of the input matrix.
    """
    dim_source = x.size(1)

    # Pad x if needed
    dd_pad = F.pad(
        x,
        pad=(0, param_dict["LL"] - dim_source, 0, 0),
        value=0,
        mode="constant",
    )

    # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
    mul_1 = torch.mul(param_dict["BB"].unsqueeze(0), dd_pad)
    # HGPi(HBX)
    mul_2 = fast_walsh_hadamard_torched_batched(mul_1, 0, normalize=False)

    # HG(PiHBX)
    mul_3 = mul_2[:, param_dict["Pi"]]

    # H(GPiHBX)
    mul_4 = torch.mul(mul_3, param_dict["GG"].unsqueeze(0))

    # (HGPiHBX)
    mul_5 = fast_walsh_hadamard_torched_batched(mul_4, 0, normalize=False)

    ret = torch.div(
        mul_5[:, :dim],
        param_dict["divisor"] * np.sqrt(float(dim) / param_dict["LL"]),
    )

    return ret


class MLP(nn.Module):
    @beartype
    def __init__(self, in_features: int, hidden_size: int, n_layers: int, out_features: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.layers.append(nn.Linear(in_features, hidden_size))
        for i in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, out_features))
        # Store initial weights and biases
        self.initial_weights = [layer.weight.clone().detach() for layer in self.layers]
        self.initial_biases = [layer.bias.clone().detach() for layer in self.layers]

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Apply the last layer without ReLU
        return x

    def weight_init(self):
        # Reset weights and biases to initial values for all layers
        for i, layer in enumerate(self.layers):
            layer.weight.data = self.initial_weights[i].clone().detach().to(layer.weight.device)
            layer.bias.data = self.initial_biases[i].clone().detach().to(layer.bias.device)
