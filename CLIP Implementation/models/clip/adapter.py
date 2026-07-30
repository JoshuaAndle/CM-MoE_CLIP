# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    """
    The experts in MoE-Adapters(++)
    Implementation by LoRA: https://arxiv.org/abs/2106.09685
    """
    def __init__(self,
                 d_model=None,  # Model dimension
                 bottleneck=None,  # Dimension of the bottleneck layer
                 dropout=0.0,  # Dropout rate
                 init_option="lora",  # Initialization option, default is "lora"
                 adapter_scalar="1.0",  # Scaling factor for the adapter, default is 1.0
                 adapter_layernorm_option="in"):  # LayerNorm option, default is "in"
        super().__init__()
        self.n_embd = d_model if d_model is None else d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, 64)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    #!# Modified intermediate activation names and output for testing/storage purposes
    def forward(self, x, add_residual=True, residual=None, verbose=False, collect_acts=False):

        residual = x if residual is None else residual

        if self.adapter_layernorm_option == 'in': #  none
            x = self.adapter_layer_norm_before(x)

        ### We return the pre-norm down and up layer activations when collecting adapter activations for metric calc
        if collect_acts == False:
            down = self.down_proj(x)
            down = self.non_linear_func(down)
            down = nn.functional.dropout(down, p=self.dropout, training=self.training)

            up = self.up_proj(down)
            up = up * self.scale

            if self.adapter_layernorm_option == 'out': #  none
                up = self.adapter_layer_norm_before(up)


            if add_residual:
                output = up + residual
            else:
                output = up

            return output

        else:
            down = self.down_proj(x)
            down_nonlinear = self.non_linear_func(down)
            down_nonlinear = nn.functional.dropout(down_nonlinear, p=self.dropout, training=self.training)

            up = self.up_proj(down_nonlinear)
            up_scaled = up * self.scale

            if self.adapter_layernorm_option == 'out': #  none
                up_scaled = self.adapter_layer_norm_before(up_scaled)

            if add_residual:
                output = up_scaled + residual
            else:
                output = up_scaled

            return (output, down, up)
