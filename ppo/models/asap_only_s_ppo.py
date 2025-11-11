from collections import deque
import torch as th
from stable_baselines3 import PPO
import traceback
from typing import Any, ClassVar, Optional, TypeVar, Union
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule, MaybeCallback
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from stable_baselines3.common.torch_layers import (
    MlpExtractor,
    create_mlp,
)

import numpy as np
import torch.nn as nn
import math
from gymnasium import spaces
from torch.nn import functional as F
from torch.func import vmap, jacrev
from functools import partial
import random
from copy import deepcopy

from stable_baselines3.common.utils import explained_variance

from scipy.fft import fft

def calculate_smoothness(actions):
    """
    주어진 액션 시퀀스의 스무스니스 지표를 계산합니다.

    :param actions: 에이전트의 액션을 담은 1차원 또는 2차원 텐서
    :return: 스무스니스 지표 값
    """
    # 액션 시퀀스가 1차원이라면 2차원으로 변환
    if actions.dim() == 1:
        actions = actions.unsqueeze(1)

    n = actions.size(0)
    if n < 2:
        return 0.0

    # 샘플링 주파수 (여기서는 1로 가정)
    fs = 1.0

    # FFT 수행 mi
    yf = th.fft.fft(actions, dim=0)
    yf = th.abs(yf[:n // 2])

    # 주파수 벡터 생성 fi
    freqs = th.fft.fftfreq(n, d=1 / fs)[:n // 2].to(yf.device).unsqueeze(1)

    # 스무스니스 계산
    smoothness = 2 * th.sum(freqs * yf, dim=0) / (n * fs)
    return smoothness.mean().item()

def calculate_oscillation(actions):
    action_n = actions[1:].float()
    action_p = actions[:-1].float()
    return th.mean(th.abs(action_n-action_p)).item()

class ASAPPolicy_nonshare(ActorCriticPolicy):
    def get_pure_action(self, obs: PyTorchObs):

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)  # (obs_dim,) -> (1, obs_dim)

        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        mean_actions = self.action_net(latent_pi)
        return mean_actions
    
    def predict_next_action(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.next_mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.next_mlp_extractor.forward_actor(pi_features)
        next_action = self.next_action_net(latent_pi)
        next_action = next_action.reshape((-1, *self.action_space.shape))
        return next_action
    
    def predict_next_action_target(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.next_mlp_extractor_target(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.next_mlp_extractor_target.forward_actor(pi_features)
        next_action = self.next_action_net_target(latent_pi)
        next_action = next_action.reshape((-1, *self.action_space.shape))
        return next_action

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        self.next_mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )


        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        
        self.next_action_net = nn.Linear(latent_dim_pi, self.action_dist.action_dim)
        # self.next_action_net = nn.Sequential(
        #     nn.Linear(latent_dim_pi, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, self.action_dist.action_dim)
        # )

        # next net target 생성
        self.next_mlp_extractor_target = deepcopy(self.next_mlp_extractor)
        self.next_action_net_target = deepcopy(self.next_action_net)
        for p in self.next_mlp_extractor_target.parameters():
            p.requires_grad = False
        for p in self.next_action_net_target.parameters():
            p.requires_grad = False

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _init_target_network(self):
        # (필요시) 학습 시작 시 한 번 완전 복사
        self.next_mlp_extractor_target.load_state_dict(
            self.next_mlp_extractor.state_dict()
        )
        self.next_action_net_target.load_state_dict(
            self.next_action_net.state_dict()
        )

    def _polyak_update_targets(self, tau: float):
        # Polyak averaging: θ_target ← (1−τ)·θ_target + τ·θ
        with th.no_grad():
            # next MLP extractor
            for p, p_targ in zip(self.next_mlp_extractor.parameters(),
                                  self.next_mlp_extractor_target.parameters()):
                p_targ.data.mul_(1.0 - tau)
                p_targ.data.add_(tau * p.data)
            # next action head
            for p, p_targ in zip(self.next_action_net.parameters(),
                                  self.next_action_net_target.parameters()):
                p_targ.data.mul_(1.0 - tau)
                p_targ.data.add_(tau * p.data)



class ASAPPolicy_share(ActorCriticPolicy):
    def get_pure_action(self, obs: PyTorchObs):

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)  # (obs_dim,) -> (1, obs_dim)

        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        mean_actions = self.action_net(latent_pi)
        return mean_actions
    
    def predict_next_action(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        next_action = self.next_action_net(latent_pi)
        next_action = next_action.reshape((-1, *self.action_space.shape))
        return next_action
    
    def predict_next_action_target(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.next_mlp_extractor_target(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.next_mlp_extractor_target.forward_actor(pi_features)
        next_action = self.next_action_net_target(latent_pi)
        next_action = next_action.reshape((-1, *self.action_space.shape))
        return next_action

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        
        self.next_action_net = nn.Linear(latent_dim_pi, self.action_dist.action_dim)
        # self.next_action_net = nn.Sequential(
        #     nn.Linear(latent_dim_pi, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, self.action_dist.action_dim)
        # )

        # next net target 생성
        self.next_mlp_extractor_target = deepcopy(self.mlp_extractor)
        self.next_action_net_target = deepcopy(self.next_action_net)
        for p in self.next_mlp_extractor_target.parameters():
            p.requires_grad = False
        for p in self.next_action_net_target.parameters():
            p.requires_grad = False

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _init_target_network(self):
        # (필요시) 학습 시작 시 한 번 완전 복사
        self.next_mlp_extractor_target.load_state_dict(
            self.mlp_extractor.state_dict()
        )
        self.next_action_net_target.load_state_dict(
            self.next_action_net.state_dict()
        )

    def _polyak_update_targets(self, tau: float):
        # Polyak averaging: θ_target ← (1−τ)·θ_target + τ·θ
        with th.no_grad():
            # next MLP extractor
            for p, p_targ in zip(self.mlp_extractor.parameters(),
                                  self.next_mlp_extractor_target.parameters()):
                p_targ.data.mul_(1.0 - tau)
                p_targ.data.add_(tau * p.data)
            # next action head
            for p, p_targ in zip(self.next_action_net.parameters(),
                                  self.next_action_net_target.parameters()):
                p_targ.data.mul_(1.0 - tau)
                p_targ.data.add_(tau * p.data)


class ASAP_ONLY_S_PPO(PPO):
    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[str, Any],
        learning_rate: Union[float, Any] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Any] = 0.2,
        clip_range_vf: Optional[Union[float, Any]] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        lam_p = 1.0,
        lam_s = 0.1,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.lam_p = lam_p
        self.lam_s = lam_s

    

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        smoothnesses = []
        oscillations = []
        predict_losses = []
        next_action_losses = []
        predict_similarities = []
        next_now_sim_compares = []
        midact_predict_compares = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                try:
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    ### predict next action
                    predicted_next_action = self.policy.predict_next_action(rollout_data.observations)[:-1]
                    next_observations = rollout_data.observations[1:]
                    target_next_action = self.policy._predict(next_observations, deterministic=True).detach()

                    predict_loss = 0.5 * F.mse_loss(predicted_next_action, target_next_action)
                    predict_losses.append(predict_loss.item())

                    ### change next action simillar with predict action
                    determ_action = self.policy._predict(next_observations, deterministic=True)
                    predicted_action =  self.policy.predict_next_action(rollout_data.observations)[:-1].detach()
                    smooth_loss = 0.5 * F.mse_loss(determ_action, predicted_action)

                    ### smoothness start
                    osc = calculate_oscillation(actions)
                    oscillations.append(osc)
                    smoothness = calculate_smoothness(actions)
                    smoothnesses.append(smoothness)

                    ### smoothness end

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.lam_p * predict_loss +  self.lam_s * smooth_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                
                except Exception as e:
                    print("🔥 예외 발생! traceback ↓↓↓")
                    traceback.print_exc()
                    raise e  # 다시 raise 하면 main thread에서도 에러 감지됨

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        
        # logging smoothness
        self.logger.record("train/smoothness", np.mean(smoothnesses))
        self.logger.record("train/oscillation", np.mean(oscillations))
        self.logger.record("train/predict_loss", np.mean(predict_losses))
        self.logger.record("train/next_action_loss", np.mean(next_action_losses))
        self.logger.record("train/predict_similarity", np.mean(predict_similarities))
        self.logger.record("train/next_now_sim_compare", np.mean(next_now_sim_compares))
        self.logger.record("train/mid_action_compare", np.mean(midact_predict_compares))
        


class ASAP_ONLY_S_PPO_L1(PPO):
    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[str, Any],
        learning_rate: Union[float, Any] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Any] = 0.2,
        clip_range_vf: Optional[Union[float, Any]] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        lam_p = 1.0,
        lam_s = 0.1,
        l1_beta = 0.1
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.lam_p = lam_p
        self.lam_s = lam_s
        self.l1_beta = l1_beta

    

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        smoothnesses = []
        oscillations = []
        predict_losses = []
        next_action_losses = []
        predict_similarities = []
        next_now_sim_compares = []
        midact_predict_compares = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                try:
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    ### predict next action
                    predicted_next_action = self.policy.predict_next_action(rollout_data.observations)[:-1]
                    next_observations = rollout_data.observations[1:]
                    target_next_action = self.policy._predict(next_observations, deterministic=True).detach()

                    predict_loss = F.smooth_l1_loss(predicted_next_action, target_next_action, beta=self.l1_beta)
                    predict_losses.append(predict_loss.item())

                    ### change next action simillar with predict action
                    determ_action = self.policy._predict(next_observations, deterministic=True)
                    predicted_action =  self.policy.predict_next_action(rollout_data.observations)[:-1].detach()
                    smooth_loss = F.smooth_l1_loss(determ_action, predicted_action, beta=self.l1_beta)

                    ### smoothness start
                    osc = calculate_oscillation(actions)
                    oscillations.append(osc)
                    smoothness = calculate_smoothness(actions)
                    smoothnesses.append(smoothness)

                    ### smoothness end

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.lam_p * predict_loss +  self.lam_s * smooth_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                
                except Exception as e:
                    print("🔥 예외 발생! traceback ↓↓↓")
                    traceback.print_exc()
                    raise e  # 다시 raise 하면 main thread에서도 에러 감지됨

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        
        # logging smoothness
        self.logger.record("train/smoothness", np.mean(smoothnesses))
        self.logger.record("train/oscillation", np.mean(oscillations))
        self.logger.record("train/predict_loss", np.mean(predict_losses))
        self.logger.record("train/next_action_loss", np.mean(next_action_losses))
        self.logger.record("train/predict_similarity", np.mean(predict_similarities))
        self.logger.record("train/next_now_sim_compare", np.mean(next_now_sim_compares))
        self.logger.record("train/mid_action_compare", np.mean(midact_predict_compares))