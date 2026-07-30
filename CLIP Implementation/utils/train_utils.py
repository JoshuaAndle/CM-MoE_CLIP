import numpy as np
import torch_optimizer
from torch.nn import Module
from torch import optim
from torch.optim import lr_scheduler


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


### Custom scheduler for split-lr setup in MoE-Adapters++
def dyn_cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def select_optimizer(opt_name: str, lr: float, model: Module) -> optim.Optimizer:

    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "adam":
        opt = optim.Adam(params, lr=lr, weight_decay=0)
    elif opt_name == "radam":
        opt = torch_optimizer.RAdam(params, lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        opt = optim.SGD(params, lr=lr, weight_decay=1e-4)
    elif opt_name == "adamw":
        opt = optim.AdamW(params, lr=lr, weight_decay=0.00001)
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    return opt


def select_scheduler(sched_name: str,
                     opt: optim.Optimizer,
                     hparam=None) -> lr_scheduler._LRScheduler:
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,
                                                                   T_0=1,
                                                                   T_mult=2)
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt,
                                                     1 / 1.1,
                                                     last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt,
                                                   milestones=[30, 60, 80, 90],
                                                   gamma=0.1)
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler




def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": classifier.state_dict()}, save_path)
    print("Checkpoint saved to", save_path)

    # with open(save_path, 'wb') as f:
    #     pickle.dump(classifier.cpu(), f)


def torch_load(classifier, save_path, device=None):
    checkpoint = torch.load(save_path)
    missing_keys, unexpected_keys = classifier.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
    print("Checkpoint loaded from", save_path)
    # with open(save_path, 'rb') as f:
    #     classifier = pickle.load(f)

    if device is not None:
        classifier = classifier.to(device)
    return classifier



# def exp_lr_scheduler(optimizer, epoch, init_lr=0.008, lr_decay_epoch=10):
def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=10):
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

