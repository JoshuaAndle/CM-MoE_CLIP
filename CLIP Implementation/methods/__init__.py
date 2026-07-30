# from .clib import CLIB
# from .er_baseline import ER
# from .ewcpp import EWCpp
# from .finetuning import FT
# from .lwf import LwF
# from .rainbow_memory import RM
# from .mvp import MVP
from .CMMoE import CMMoE
from .ZSCL import ZSCL
from .DAC import DAC
from .DIKI import DIKI
from .moe_adapters_pp import moe_adapters_pp
from .continual_clip import ContinualCLIP

__all__ = [
    # "CLIB",
    # "ER",
    # "EWCpp",
    # "FT",
    # "LwF",
    # "RM",
    # "MVP",
    "CMMoE",
    "DAC",
    "DIKI",
    "ZSCL",
    "moe_adapters_pp",
    "continual_clip",
]

def get_method(name):
    name = name.lower()
    # try:
    #     return {
    #         "clib": CLIB,
    #         "er": ER,
    #         "ewcpp": EWCpp,
    #         "ft": FT,
    #         "lwf": LwF,
    #         "rm": RM,
    #         "mvp": MVP,
    #         "cmmoe": CMMoE,
    #         "moe_adapters_pp": moe_adapters_pp,
    #         "continual_clip": continual_clip,
    #     }[name]
    # except KeyError:
    raise NotImplementedError(f"Method {name} not implemented")