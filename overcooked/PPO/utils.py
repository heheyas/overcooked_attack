import torch as th
from .model import CustomCNN
from a2c_ppo_acktr.model import Policy

def load_ac_from_file(params):
    return th.load(params["trained_ac_path"])