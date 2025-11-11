# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--checkpath", type=str, default=None, help="Path to model checkpoint dir.")
parser.add_argument("--seedpath", type=str, default=None, help="Path to test seed.")
parser.add_argument("--maxloop", type=int, default=10, help="Number to max test loop")
parser.add_argument("--csv_name", type=str, default="default", help="csv file name")
parser.add_argument("--alg_name", type=str, default="base", help="algorithm name")
parser.add_argument("--noise_mul", type=float, default=1.0, help="multiple noise")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os, glob, re
import time
import torch
import numpy as np
import yaml
import csv

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
from rl_games.algos_torch import players

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.asap_ppo import A2C_ASAPAgent, ASAPBuilder, ModelASAPContinuous, ModelASAPContinuousLogStd, PpoPlayerASAPContinuous

_orig_torch_load = torch.load

# (3) map_location 디폴트를 cuda:0 으로 설정하는 래퍼
def _patched_torch_load(*args, **kwargs):
    # 만약 map_location이 지정되지 않았다면 자동으로 cuda:0(또는 'cpu') 로 보냄
    device_arg = args_cli.device if args_cli.device is not None else "cuda:0"
    kwargs.setdefault("map_location", device_arg)
    return _orig_torch_load(*args, **kwargs)

# (4) 덮어쓰기
torch.load = _patched_torch_load

def extract_epoch(path):
    fname = os.path.basename(path)
    m = re.search(r"ep_(\d+)", fname)
    return int(m.group(1)) if m else -1

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def calculate_smoothness_np(actions: np.ndarray, fs: float = 1.0) -> float:
    """
    NumPy 로 구현한 smoothness 지표.
    actions: shape (T,) 또는 (T, d)  (T = timestep 수, d = 액션 차원)
    fs: 샘플링 주파수 (기본 1.0)
    """
    # 1차원일 때 (T,) -> (T,1)
    a = np.array(actions, dtype=float)
    if a.ndim == 1:
        a = a[:, None]

    n = a.shape[0]
    if n < 2:
        return 0.0

    # FFT
    # axis=0 방향으로 fft, 양의 주파수 절반만 취함
    yf = np.fft.fft(a, axis=0)
    yf = np.abs(yf[: n // 2, :])    # shape (n//2, d)

    # 주파수 벡터 생성 (n//2 길이)
    freqs = np.fft.fftfreq(n, d=1/fs)[: n // 2]  # shape (n//2,)
    freqs = freqs.reshape(-1, 1)                  # (n//2,1)

    # 식: Sm = 2/(n*fs) * sum_i (M_i * f_i)
    smooth_per_dim = (2.0 / (n * fs)) * np.sum(freqs * yf, axis=0)  # shape (d,)

    # 다차원 액션이면 차원별 평균, 1차원 액션이면 그냥 원소 반환
    return float(np.mean(smooth_per_dim))

def csv_writer(re_mean, re_std, sm_mean, sm_std, task_name, al_name="base"):
    basic_envs = ["al_name", task_name]

    rows = [basic_envs]

    re_mean_row = ["re_mean", re_mean]
    re_std_row = ["re_std", re_std]
    sm_mean_row = ["sm_mean", sm_mean]
    sm_std_row = ["sm_std", sm_std]

    rows.append(re_mean_row)
    rows.append(re_std_row)
    rows.append(sm_mean_row)
    rows.append(sm_std_row)

    cwd = os.getcwd()
    subdir = f"{task_name}_{al_name}"
    combined_path = os.path.join(cwd, subdir)
    os.makedirs(combined_path, exist_ok=True)

    with open(os.path.join(combined_path, f"{subdir}.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 3) 한 줄씩 쓰기
        for row in rows:
            writer.writerow(row)

def main():
    """Play with RL-Games agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    # if args_cli.use_pretrained_checkpoint:
    #     resume_path = get_published_pretrained_checkpoint("rl_games", task_name)
    #     if not resume_path:
    #         print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
    #         return
    # elif args_cli.checkpoint is None:
    #     # specify directory for logging runs
    #     run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
    #     # specify name of checkpoint
    #     if args_cli.use_last_checkpoint:
    #         checkpoint_file = ".*"
    #     else:
    #         # this loads the best checkpoint
    #         checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
    #     # get path to previous checkpoint
    #     resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    # else:
    #     resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = "log_dir"

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # noise multiple
    if args_cli.task == "Isaac-Repose-Cube-Allegro-v0":
        print(f"task : Isaac-Repose-Cube-Allegro-v0. noise multiple : {args_cli.noise_mul}")
        noise_mul = args_cli.noise_mul
        print(f"joint_pos_noise : {env_cfg.observations.policy.joint_pos.noise.std}")
        env_cfg.observations.policy.joint_pos.noise.std = env_cfg.observations.policy.joint_pos.noise.std * noise_mul
        print(f"joint_pos_noise : {env_cfg.observations.policy.joint_pos.noise.std}")
        print(f"joint_vel_noise : {env_cfg.observations.policy.joint_vel.noise.std}")
        env_cfg.observations.policy.joint_vel.noise.std = env_cfg.observations.policy.joint_vel.noise.std * noise_mul
        print(f"joint_vel_noise : {env_cfg.observations.policy.joint_vel.noise.std}")
        env_cfg.observations.policy.object_pos.noise.std = env_cfg.observations.policy.object_pos.noise.std * noise_mul
        env_cfg.observations.policy.object_lin_vel.noise.std = env_cfg.observations.policy.object_lin_vel.noise.std * noise_mul
        env_cfg.observations.policy.object_ang_vel.noise.std = env_cfg.observations.policy.object_ang_vel.noise.std * noise_mul

    if args_cli.task == "Isaac-Velocity-Rough-Anymal-C-v0":
        print(f"task : Isaac-Velocity-Rough-Anymal-C-v0. noise multiple : {args_cli.noise_mul}")
        noise_mul = args_cli.noise_mul
        print(f"base_lin_vel_noise : {env_cfg.observations.policy.base_lin_vel.noise.n_min}")
        env_cfg.observations.policy.base_lin_vel.noise.n_min = env_cfg.observations.policy.base_lin_vel.noise.n_min * noise_mul
        env_cfg.observations.policy.base_lin_vel.noise.n_max = env_cfg.observations.policy.base_lin_vel.noise.n_max * noise_mul
        print(f"base_lin_vel_noise : {env_cfg.observations.policy.base_lin_vel.noise.n_min}")
        env_cfg.observations.policy.base_ang_vel.noise.n_min = env_cfg.observations.policy.base_ang_vel.noise.n_min * noise_mul
        env_cfg.observations.policy.base_ang_vel.noise.n_max = env_cfg.observations.policy.base_ang_vel.noise.n_max * noise_mul

        env_cfg.observations.policy.projected_gravity.noise.n_min = env_cfg.observations.policy.projected_gravity.noise.n_min * noise_mul
        env_cfg.observations.policy.projected_gravity.noise.n_max = env_cfg.observations.policy.projected_gravity.noise.n_max * noise_mul

        env_cfg.observations.policy.joint_pos.noise.n_min = env_cfg.observations.policy.joint_pos.noise.n_min * noise_mul
        env_cfg.observations.policy.joint_pos.noise.n_max = env_cfg.observations.policy.joint_pos.noise.n_max * noise_mul

        env_cfg.observations.policy.joint_vel.noise.n_min = env_cfg.observations.policy.joint_vel.noise.n_min * noise_mul
        env_cfg.observations.policy.joint_vel.noise.n_max = env_cfg.observations.policy.joint_vel.noise.n_max * noise_mul

        env_cfg.observations.policy.height_scan.noise.n_min = env_cfg.observations.policy.height_scan.noise.n_min * noise_mul
        env_cfg.observations.policy.height_scan.noise.n_max = env_cfg.observations.policy.height_scan.noise.n_max * noise_mul

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    #### 여기부터 for로 묶어서 할거임ㅇㅇ
    seeds = load_seeds(args_cli.seedpath)
    maxloop = args_cli.maxloop
    counter = 0
    pth_dict = {}
    pth_root_path = args_cli.checkpath
    pth_folders_rew = os.listdir(pth_root_path)  
    pth_folders = [os.path.join(pth_root_path, d) for d in pth_folders_rew if os.path.isdir(os.path.join(pth_root_path, d))]

    for pth_folder in pth_folders:
        nn = os.path.join(pth_folder, "nn")
        nn_list = glob.glob(os.path.join(nn, "*.pth"))
        if not nn_list:
            continue
        latest_ckpt = max(nn_list, key=extract_epoch)

        param_file = os.path.join(pth_folder, "params","agent.yaml")
        pth_seed = 0
        with open(param_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            pth_seed = config["params"]["seed"]
        pth_dict[pth_seed] = latest_ckpt

    reward_list = []
    smoothness_list = []

    if args_cli.alg_name == "asap":
        agent_cfg["params"]["algo"]["name"] = "asap_continuous"
        agent_cfg["params"]["network"]["name"] = "actor_critic_asap"
        if agent_cfg["params"]["model"]["name"] == "continuous_a2c":
            agent_cfg["params"]["model"]["name"] = "continuous_asap"
        if agent_cfg["params"]["model"]["name"] == "continuous_a2c_logstd":
            agent_cfg["params"]["model"]["name"] = "continuous_asap_logstd"


    for pth_seed in sorted(pth_dict):
        # load previously trained model
        pth_each_path = pth_dict[pth_seed]
        resume_path = pth_each_path
        print(f"pth_seed : {pth_seed}")
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

        # set number of actors into agent config
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        # create runner from rl-games
        runner = Runner()
        runner.algo_factory.register_builder(
            'asap_continuous',
            lambda **kwargs: A2C_ASAPAgent(**kwargs)
        )
        runner.player_factory.register_builder('asap_continuous', lambda **kwargs : PpoPlayerASAPContinuous(**kwargs))
        runner.load(agent_cfg)
        # obtain the agent from the runner
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

        dt = env.unwrapped.step_dt

        # simulate environment
        # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
        #   attempt to have complete control over environment stepping. However, this removes other
        #   operations such as masking that is used for multi-agent learning by RL-Games.
        for loop_num in range(maxloop):
            episode_return = 0.0
            action_list = []

            env.seed(seeds[counter])
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            timestep = 0
            # required: enables the flag for batched observations
            _ = agent.get_batch_size(obs, 1)
            # initialize RNN states if used
            if agent.is_rnn:
                agent.init_rnn()

            episode_done = False

            while simulation_app.is_running() and not episode_done:
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # convert obs to agent format
                    obs = agent.obs_to_torch(obs)
                    # agent stepping
                    # actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
                    actions = agent.get_action(obs, is_deterministic=False)
                    # env stepping
                obs, rewards, dones, _ = env.step(actions)

                r = rewards.item()
                episode_return += r

                a = actions.cpu().numpy()[0]   # shape (d,)
                action_list.append(a)

                if any(dones):
                    episode_done = True
                    # reset rnn state for terminated episodes
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0
                if args_cli.video:
                    timestep += 1
                    # exit the play loop after recording one video
                    if timestep == args_cli.video_length:
                        break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
            counter += 1

            actions_arr = np.stack(action_list, axis=0)
            smoothness = calculate_smoothness_np(actions_arr, fs=1.0)

            reward_list.append(episode_return)
            smoothness_list.append(smoothness)
            print(f"re : {episode_return}")
            print(f"smooth : {smoothness}")

    smoothness_avg = float(np.mean(smoothness_list))
    smoothness_std = float(np.std(smoothness_list))
    reward_avg = float(np.mean(reward_list))
    reward_std = float(np.std(reward_list))

    csv_writer(reward_avg, reward_std, smoothness_avg, smoothness_std, task_name, args_cli.csv_name)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
