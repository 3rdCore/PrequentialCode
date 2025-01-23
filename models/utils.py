from typing import Any, Optional

import numpy as np
import torch
from beartype import beartype
from torch import Tensor, nn
from torch.nn import functional as F


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


class FastFoodUpsample(nn.Module):
    def __init__(self, low_dim: int, param_shape: tuple[int, ...]):
        super().__init__()
        self.low_dim = low_dim
        self.param_shape = param_shape
        self.total_dim = np.prod(self.param_shape)

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
        delta = delta.view(-1, *self.param_shape)
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
    def __init__(
        self,
        in_features: int,
        h_dim: int,
        n_layers: int,
        out_features: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.layers.append(nn.Linear(in_features, h_dim))
        for i in range(n_layers - 2):
            self.layers.append(nn.Linear(h_dim, h_dim))
        self.layers.append(nn.Linear(h_dim, out_features))
        # Store initial weights and biases
        self.initial_weights = [layer.weight.clone().detach() for layer in self.layers]
        self.initial_biases = [layer.bias.clone().detach() for layer in self.layers]

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)  # Apply the last layer without ReLU
        return x

    def weight_init(self):
        # Reset weights and biases to initial values for all layers
        for i, layer in enumerate(self.layers):
            layer.weight.data = self.initial_weights[i].clone().detach().to(layer.weight.device)
            layer.bias.data = self.initial_biases[i].clone().detach().to(layer.bias.device)


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.silu,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else int(8 * in_features / 3)
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y


class RNNSeq2Seq(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        x_len: int,
        y_len: int,
        h_dim: int = 128,
        nonlinearity: str = "tanh",
        **kwargs,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.x_len = x_len
        self.y_len = y_len

        self.encoder = nn.RNN(
            dim,
            h_dim,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        self.decoder = nn.RNN(
            dim,
            h_dim,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        self.pred = nn.Linear(h_dim, dim)

        self.initial_params = {n: v.clone().detach() for n, v in self.named_parameters()}

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, self.x_len, self.dim)
        _, h = self.encoder.forward(x)
        input = torch.zeros(batch_size, self.y_len, self.dim).to(x.device)
        y, _ = self.decoder.forward(input, h)
        y = self.pred(y)
        return y.view(batch_size, -1)

    def weight_init(self):
        for n, p in self.named_parameters():
            p.data = self.initial_params[n].clone().detach().to(p.device)


class VectorField(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        y_len: int,
        nonlinearity: Any = nn.Identity(),
        dt: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.y_len = y_len
        self.nonlinearity = nonlinearity
        self.dt = dt

        self.w = nn.Parameter(torch.randn(dim, dim))

        self.initial_w = self.w.clone().detach()

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, -self.dim :]  # Only current state matters
        y = []
        for _ in range(self.y_len):
            x = x + self.dt * (x @ self.w)
            x = self.nonlinearity(x)
            y.append(x)
        y = torch.cat(y, dim=1)
        return y

    def weight_init(self):
        self.w.data = self.initial_w.clone().detach().to(self.w.device)


class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal
            )
            memory = memory + self._mha_block(
                self.norm2(memory), memory, memory_mask, memory_key_padding_mask, memory_is_causal
            )

            x = x + self._ff_block(self.norm3(x))
            memory = memory + self._ff_block(self.norm3(memory))
        else:
            x = self.norm2(
                x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            )
            memory = self.norm2(
                memory
                + self._mha_block(memory, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            )
            x = self.norm3(x + self._ff_block(x))
            memory = self.norm3(memory + self._ff_block(memory))

        return x, memory

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        # TODO: Update mem using self_attn
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)


class CustomDecoder(nn.TransformerDecoder):
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output, memory = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class CustomDecoderLayer2(CustomDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(CustomDecoderLayer2, self).__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            device,
            dtype,
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            device,
            dtype,
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        x, _ = super().forward(
            x,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            tgt_is_causal,
            memory_is_causal,
        )
        memory = self.encoder_layer(memory, memory_mask, memory_key_padding_mask)
        return x, memory
