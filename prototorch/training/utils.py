import numpy as np
import torch
import json
import yaml


def get_device(force_cpu=False):
    if torch.cuda.is_available() and not force_cpu:
        # works in multi-gpu
        # https://discuss.pytorch.org/t/difference-between-cuda-0-vs-cuda-with-1-gpu/93080
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device", device, flush=True)
    return device


def get_amp_dtype(amp_type):
    return torch.bfloat16 if amp_type == "bfloat16" else torch.float16


def isnumeric(v):
    """Required because we support non numeric metrics for wandb."""
    return isinstance(v, float) or isinstance(v, int) or np.isscalar(v)


def set_seeds(seed):
    # TODO for ddp we may want to set PYTHONHASHSEED=0
    # on parameter initialisation in ddp
    # https://discuss.pytorch.org/t/setting-seed-in-torch-ddp/126638
    if seed is not None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        # also see pytorch lightning seed everything
        print(f"seeding {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)


def load_yaml(yaml_file):
    # * Make FullLoader safer by removing python/object/apply from the default FullLoader
    # https://github.com/yaml/pyyaml/pull/347
    # Move constructor for object/apply to UnsafeConstructor
    with open(yaml_file, "r") as yf:
        return yaml.load(yf, Loader=yaml.UnsafeLoader)


def load_json(jsonfile):
    with open(jsonfile, "r") as jf:
        res = json.load(jf)
    return res
