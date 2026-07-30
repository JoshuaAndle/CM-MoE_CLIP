# import torch
import time
from configuration import config
from OCL_datasets import *
from methods.continual_clip import ContinualCLIP
from methods.moe_adapters_pp import moe_adapters_pp
from methods.ZSCL import ZSCL
from methods.DAC import DAC
from methods.DIKI import DIKI
from methods.CMMoE import CMMoE

# torch.backends.cudnn.enabled = False
methods = {
    "CMMoE": CMMoE,
    "ZSCL": ZSCL,
    "DAC": DAC,
    "DIKI": DIKI,
    "moe_adapters_pp": moe_adapters_pp,
    "continual_clip": ContinualCLIP,
}


def main():
    # Get Configurations
    args = config.base_parser()
    # trainer = methods[args.method](args, **vars(args))
    trainer = methods[args.method](args)

    trainer.run()


if __name__ == "__main__":
    main()
