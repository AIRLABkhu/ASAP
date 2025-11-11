import sys
import gymnasium as gym
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse
import signal
from torch import nn

from stable_baselines3.common.env_util import make_vec_env

from modules.action_extractor import test_some_path
from modules.controller import train_grad_cs_param
from modules.envs import make_ant_env, make_hopper_env, make_humanoid_env, make_lunar_env, make_pendulum_env, make_reacher_env, make_walker_env

from modules.params import env_timestep, env_args, alg_args

train_envs_dict = dict({
    "ant" : make_ant_env,
    "hopper" : make_hopper_env,
    "humanoid" : make_humanoid_env,
    "lunar" : make_lunar_env,
    "pendulum" : make_pendulum_env,
    "reacher" : make_reacher_env,
    "walker" : make_walker_env
})

alg_cnts = dict({
    "grad_cs" : train_grad_cs_param
})


save_dir_root = f"./ppo/results/pths/"
logs_dir_root = f"./ppo/results/tensorboard_logs/"
seed_file_path = "./ppo/tests/seeds.txt"
result_dir_root = f"./ppo/results/"
max_num = 10
max_concurrent_num = 5

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]
    
def mk_param_list_lam(lam_s_list, lam_t_list):
    param_list = []
    pth_list=[]
    for lam_t in lam_t_list:
        for lam_s in lam_s_list:
            param_list.append(dict(
                grad_sigma = 0.2,
                grad_lamS = lam_s,
                grad_lamT = lam_t))
            pth_list.append(f"grad_cs_ppo_lamS{lam_s}_lamT{lam_t}_")
    return param_list, pth_list

## 학습 시작

parser = argparse.ArgumentParser()
parser.add_argument(
    "--envs",
    nargs="+",
    choices=["ant", "hopper", "lunar", "pendulum", "reacher", "walker"],
    default=["ant", "hopper", "lunar", "pendulum", "reacher", "walker"],
    help="List of environments to train on."
)
parser.add_argument(
    "--lams",
    nargs="+",
    type=float,
    default=[0.5, 0.3],
    help="List of lambda values (floats). 예: --lams 0.1 1.5 2.3"
)

parser.add_argument(
    "--lamt",           # 옵션 이름
    nargs="+",
    type=float,
    default=[0.1, 0.05],
    help="List of lambda values (floats). 예: --lams 0.1 1.5 2.3"
)

parser.add_argument(
    "--num",           # 옵션 이름
    type=int,        # 타입은 float
    default=10,       # 기본값
    help="test num for each parameter (int). 예: --num 3"
)
parser.add_argument(
    "--cnum",           # 옵션 이름
    type=int,        # 타입은 float
    default=3,       # 기본값
    help="test num for each parameter (int). 예: --num 3"
)
args = parser.parse_args()
print(f"Selected envs: {args.envs}")
print(f"Selected max_num: {args.num}")

# 테스트할 컨트롤러 목록
train_algs = ["grad_cs"]
# 테스트할 env 목록
train_envs = args.envs #["ant", "hopper", "humanoid", "lunar", "pendulum", "reacher"]
test_params, pth_list = mk_param_list_lam(args.lams, args.lamt)
max_num = args.num
max_concurrent_num = args.cnum

# 한개 알고리즘을 1개 env에서 테스트하는데 약 2시간 소요
# -> 모든 env 테스트시 12시간, 모든 알고리즘 테스트시 약 60시간

seeds = load_seeds(seed_file_path)
if len(seeds) < max_num:
    raise ValueError(f"Not enough seeds in {seed_file_path}: required {max_num}, found {len(seeds)}")

jobs = []
for env_name in train_envs:

    save_dir = os.path.join(save_dir_root, env_name) + '/'
    log_dir = os.path.join(logs_dir_root, env_name) + '/'


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        time.sleep(2)

    for num in range(max_num):
        seed = seeds[num]
        for alg_name in train_algs:
            for test_param in test_params:
                jobs.extend([
                    partial(alg_cnts[alg_name], seed, env_timestep[env_name], save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], test_param),
                ])

with ProcessPoolExecutor(max_workers=max_concurrent_num) as executor:
    future_to_job = {executor.submit(job): job for job in jobs}

    for future in as_completed(future_to_job):
        job_func = future_to_job[future]
        try:
            future.result()
        except Exception as exc:
            print(f"Job {job_func} generated an exception: {exc}")
        else:
            print(f"Job {job_func} completed successfully.")

subpath = f"grad_cs_{args.envs[0]}_lamt{args.lamt[0]}/"
test_some_path(save_dir_root, True, pth_list, subpath)