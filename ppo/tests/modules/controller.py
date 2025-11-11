import sys
import gymnasium as gym
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Callable, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.custom_ppo import CustomPPO, L2C2PPO, CAPSPPO, LipsPPO, CAPSTPPO
from models.custom_policy import LipsPolicy
from models.grad_ppo import GRADPPO, GRAD_CS_PPO, GRAD_LS_PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from models.asap_only_s_ppo import ASAP_ONLY_S_PPO, ASAPPolicy_share
from models.asap_ppo import ASAPPPO, CAPST_ASAPPPO, ASAPPPO_lam_undetach

def train_vanilla(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model = CustomPPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, 
                      seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"PPO_{seed}")
    model.save(f"{save_dir}vanilla_{seed}")
    vec_env.close()
    del model

def train_caps(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model = CAPSPPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir,
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"CAPS_PPO_{seed}")
    model.save(f"{save_dir}caps_ppo_{seed}")
    vec_env.close()
    del model

def train_l2c2(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model = L2C2PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir,
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"L2C2PPO_{seed}")
    model.save(f"{save_dir}l2c2_ppo_{seed}")
    vec_env.close()
    del model

def train_lips(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, pargs: Optional[dict] = None):
    if pargs is None:
        pargs = {
            "lips_lam": 1e-5
        }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vec_env = mkenv_func()
    model = LipsPPO(LipsPolicy, vec_env, verbose=0, tensorboard_log=log_dir,
                    lips_lam=pargs["lips_lam"], lips_eps=1e-4, lips_k_init=1, seed=seed)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"Lips_PPO_{seed}")
    model.save(f"{save_dir}lips_ppo_{seed}")
    vec_env.close()
    del model


def train_asap_only_s_share(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
              env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model = ASAP_ONLY_S_PPO(ASAPPolicy_share, vec_env, verbose=0, tensorboard_log=log_dir, 
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"ASAP_ONLY_S_PPO_{seed}")
    model.save(f"{save_dir}asap_only_s_ppo_{seed}")
    vec_env.close()
    del model

def train_asap_only_s_share_param(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
              env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model = ASAP_ONLY_S_PPO(ASAPPolicy_share, vec_env, verbose=0, tensorboard_log=log_dir, 
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"ASAP_ONLY_S_PPO_lamS{alg_args['lam_s']}_lamP{alg_args['lam_p']}_{seed}")
    model.save(f"{save_dir}asap_only_s_ppo_lamS{alg_args['lam_s']}_lamP{alg_args['lam_p']}_{seed}")
    vec_env.close()
    del model

def train_grad(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model = GRADPPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir,
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"GRAD_PPO_{seed}")
    model.save(f"{save_dir}grad_ppo_{seed}")
    vec_env.close()
    del model

def train_grad_param(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model = GRADPPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir,
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"GRAD_PPO_lamT{alg_args['grad_lamT']}_{seed}")
    model.save(f"{save_dir}grad_ppo_lamT{alg_args['grad_lamT']}_{seed}")
    vec_env.close()
    del model

def train_capst_asap(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
              env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model = CAPST_ASAPPPO(ASAPPolicy_share, vec_env, verbose=0, tensorboard_log=log_dir, 
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"CAPST_ASAP_PPO_{seed}")
    model.save(f"{save_dir}capst_asap_ppo_{seed}")
    vec_env.close()
    del model

def train_capst_asap_param(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
              env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model =CAPST_ASAPPPO(ASAPPolicy_share, vec_env, verbose=0, tensorboard_log=log_dir, 
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"CAPST_ASAP_PPO_lamS{alg_args['lam_s']}_lamT{alg_args['lam_t']}_{seed}")
    model.save(f"{save_dir}capst_asap_ppo_lamS{alg_args['lam_s']}_lamT{alg_args['lam_t']}_{seed}")
    vec_env.close()
    del model


def train_asap(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
              env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model = ASAPPPO(ASAPPolicy_share, vec_env, verbose=0, tensorboard_log=log_dir, 
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"ASAP_PPO_{seed}")
    model.save(f"{save_dir}asap_ppo_{seed}")
    vec_env.close()
    del model

def train_asap_param(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
              env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model =ASAPPPO(ASAPPolicy_share, vec_env, verbose=0, tensorboard_log=log_dir, 
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"ASAP_PPO_lamS{alg_args['lam_s']}_lamT{alg_args['lam_t']}_lamP{alg_args['lam_p']}_{seed}")
    model.save(f"{save_dir}asap_ppo_lamS{alg_args['lam_s']}_lamT{alg_args['lam_t']}_lamP{alg_args['lam_p']}_{seed}")
    vec_env.close()
    del model


def train_grad_cs_param(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
              env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model =GRAD_CS_PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, 
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"GRAD_CS_PPO_lamS{alg_args['grad_lamS']}_lamT{alg_args['grad_lamT']}_{seed}")
    model.save(f"{save_dir}grad_cs_ppo_lamS{alg_args['grad_lamS']}_lamT{alg_args['grad_lamT']}_{seed}")
    vec_env.close()
    del model

def train_grad_ls_param(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
              env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model =GRAD_LS_PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, 
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"GRAD_LS_PPO_lamD{alg_args['grad_lamD']}_lamT{alg_args['grad_lamT']}_{seed}")
    model.save(f"{save_dir}grad_ls_ppo_lamD{alg_args['grad_lamD']}_lamT{alg_args['grad_lamT']}_{seed}")
    vec_env.close()
    del model


def train_caps_t_param(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
              env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model =CAPSTPPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, 
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"CAPS_T_PPO_lamT{alg_args['caps_lamT']}_{seed}")
    model.save(f"{save_dir}caps_t_ppo_lamT{alg_args['caps_lamT']}_{seed}")
    vec_env.close()
    del model


def train_asap_undetach_param(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
              env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    ppo_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    ppo_args.update(alg_args)
    model =ASAPPPO_lam_undetach(ASAPPolicy_share, vec_env, verbose=0, tensorboard_log=log_dir, 
                    seed=seed, **ppo_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"ASAP_UNDETACH_PPO_lamS{alg_args['lam_s']}_lamT{alg_args['lam_t']}_{seed}")
    model.save(f"{save_dir}asap_undetach_ppo_lamS{alg_args['lam_s']}_lamT{alg_args['lam_t']}_{seed}")
    vec_env.close()
    del model