
from .CMMoE import CMMoE
from .DAC import DAC
from .DIKI import DIKI
from .ZSCL import ZSCL
from .continual_clip import ContinualCLIP
from .moe_adapters_pp import moe_adapters_pp


def get_model(method, args, model_name, **kwargs):

    if method == "continual_clip":
        return ContinualCLIP(args=args,
                            model_name=model_name,
                            device=kwargs['device']), 224
    elif method == "CMMoE":
        return CMMoE(args=args,
                      model_name=model_name,
                      device=kwargs['device'],
                      log_dir=kwargs['log_dir']), 224
    elif method == "ZSCL":
        return ZSCL(args=args,
                    model_name=model_name,
                    device=kwargs['device'],
                    log_dir=kwargs['log_dir']), 224
    elif method == "DAC":
        return DAC(args=args,
                    model_name=model_name,
                    device=kwargs['device'],
                    log_dir=kwargs['log_dir']), 224        
    elif method == "DIKI":
        return DIKI(args=args,
                    model_name=model_name,
                    device=kwargs['device'],
                    log_dir=kwargs['log_dir']), 224        
    elif method == "moe_adapters_pp":
        return moe_adapters_pp(args=args,
                                model_name=model_name,
                                device=kwargs['device'],
                                peft_method='lora',
                                peft_encoder=kwargs['peft_encoder'],
                                log_dir=kwargs['log_dir']), 224

    else:
        raise NotImplementedError(
            f"Model {method}_{model_name} not implemented")
