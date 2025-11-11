# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--lams", type=float, default=0.3, help="lamda spatial")
parser.add_argument("--lamp", type=float, default=1.0, help="lamda predict")
parser.add_argument("--lamt", type=float, default=0.05, help="lamda temporal")
parser.add_argument("--noise_mul", type=float, default=1.0, help="multiple noise")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.asap_ppo import A2C_ASAPAgent, ASAPBuilder, ModelASAPContinuous, ModelASAPContinuousLogStd, PpoPlayerASAPContinuous

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

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

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
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

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # asap로 설정 바꾸기
    agent_cfg["params"]["algo"]["name"] = "asap_continuous"
    agent_cfg["params"]["network"]["name"] = "actor_critic_asap"
    if agent_cfg["params"]["model"]["name"] == "continuous_a2c":
        agent_cfg["params"]["model"]["name"] = "continuous_asap"
    if agent_cfg["params"]["model"]["name"] == "continuous_a2c_logstd":
        agent_cfg["params"]["model"]["name"] = "continuous_asap_logstd"
    # asap config 추가하기
    agent_cfg["params"]["config"]["lam_spatial"] = args_cli.lams
    agent_cfg["params"]["config"]["lam_predict"] = args_cli.lamp
    agent_cfg["params"]["config"]["lam_temporal"] = args_cli.lamt

    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.algo_factory.register_builder(
        'asap_continuous',
        lambda **kwargs: A2C_ASAPAgent(**kwargs)
    )
    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent

    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        wandb.config.update({"env_cfg": env_cfg.to_dict()})
        wandb.config.update({"agent_cfg": agent_cfg})

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
