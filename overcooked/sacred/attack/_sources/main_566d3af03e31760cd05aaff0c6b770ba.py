import os
import sys
from syslog import LOG_PID

from sklearn.decomposition import TruncatedSVD
sys.path.append("..")
import time
from collections import deque
import torch
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor, configure_logger
from a2c_ppo_acktr import utils
from envior.env import OvercookedMultiAgent
from pathlib import Path
import wandb
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from envior.env_utils import get_vectorized_gym_env, linear_anealing
from PPO import model
from PPO.Trainer import AdvTrainer

ex = Experiment("attack")
ex.observers.append(FileStorageObserver("./sacred/attack"))
debug = False

@ex.config
def config():
    run_type = "ppo"
    other_agent = "sp"
    # layout_name = "counter_circuit_o_1order"
    layout_name = "cramped_room"
    sim_threads = 30
    
    seed = 100
    
    total_timesteps = 1e7
    
    total_batch_size = 12000
    
    n_epochs = 8
    
    deterministic = False
    clip_range = .05
    lr = 1e-3
    ent_coef = .01
    vf_coef = .05 
    gamma = .99
    gae_lambda = .98
    use_gae = True
    max_grad_norm = .1
    
    save_interval = 5
    log_interval = 1

    #################
    use_compat = False
    use_rnn = False
    hidden_size = 32
    
    num_mini_batch = 6
    mini_batch_size = total_batch_size // sim_threads
    
    reward_shaping_horizon = int(2.5e6)
    reward_shaping_factor = 1.
    
    horizon = 400
    
    bc_schedule = OvercookedMultiAgent.bc_schedule
    
    device = "cuda:0"
    
    use_phi = False
    
    # lr_schedule_kwargs = dict(
    #     threshold=.3,
    #     start=0,
    #     end=total_timesteps,
    # )
    lr_schedule_kwargs = None
    
    reward_shaping_kwargs = dict(
        threshold=0,
        start=0,
        end=reward_shaping_horizon,
    )
    
    verbose = 0
    tensorboard_log = "./log/sp/"
    tb_log_name = f"{layout_name}_"
    
    save_dir = f"../saved_models/sp"
    if use_rnn:
        save_dir += "_recurr"
    if use_compat:
        save_dir += "_compat"
    
    save_dir = Path(save_dir) / layout_name
    
    trained_ac_path = "/opt/czl/self/new_overcooked_apag/saved_models/sp/cramped_room/policy.pt"
    bootstrapped = True
    
    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }

    env_config = {
        "mdp_params" : {
            "layout_name": layout_name,
            "rew_shaping_params": rew_shaping_params
        },
        # To be passed into OvercookedEnv constructor
        "env_params" : {
            "horizon" : horizon
        },

        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params" : {
            "reward_shaping_factor" : reward_shaping_factor,
            "reward_shaping_horizon" : reward_shaping_horizon,
            "use_phi" : use_phi,
            "bc_schedule" : bc_schedule
        }
    }
    
    params = {
        "run_type": run_type,    
        "use_compat": use_compat,
        "seed": seed,
        "layout_name": layout_name,
        "use_rnn": use_rnn,
        "hidden_size": hidden_size,
        "env_config": env_config,
        "device": torch.device(device),
        "clip_range": clip_range,
        "deterministic": deterministic,
        "n_epochs": n_epochs,
        "num_mini_batch": num_mini_batch, 
        "mini_batch_size": mini_batch_size,
        "vf_coef": vf_coef, 
        "ent_coef": ent_coef, 
        "lr": lr,
        "eps": 1e-8,
        "max_grad_norm": max_grad_norm, 
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "use_gae": use_gae,
        "total_batch_size": total_batch_size,
        "sim_threads": sim_threads,
        "total_timesteps": total_timesteps,
        "save_interval": save_interval, 
        "save_dir": save_dir,
        "log_interval": log_interval,
        "reward_shaping_horizon": reward_shaping_horizon,
        "lr_schedule_kwargs": lr_schedule_kwargs,
        "reward_shaping_kwargs": reward_shaping_kwargs,
        "other_agent": other_agent,
        
        "verbose": verbose,
        "tensorboard_log": tensorboard_log,
        "tb_log_name": tb_log_name,
        
        "bootstrapped": bootstrapped,
        "trained_ac_path": trained_ac_path,
    }
    

@ex.command
def config_to_yaml(params):
    import yaml
    config = {}
    config["env"] = "overcooked"
    config["env_args"] = params["env_config"]
    with open("overcooked.yaml", "w") as f:
        f.write(yaml.dump(config, allow_unicode=True))
        
    
@ex.automain
def run(params, _run):
    tags = ["sp", "ppo", f"{params['layout_name']}"]
    name = f"ppo_sp_{params['layout_name']}_{_run._id}"
    
    if params["use_rnn"]:
        tags.append("rnn")
        name += "_rnn"
    
    if params["use_compat"]:
        tags.append("use_compat")
        name += "_compat"
    print(params["layout_name"])
        
    if not debug:
        wandb.init(project="overcooked-attack", sync_tensorboard=True, config=params, name=name, tags=tags)
    params["tb_log_name"] += f"{_run._id}"
    # ott = lambda x: obs_as_tensor(x, params["device"])
    trainer = AdvTrainer(params)
    trainer.update()
