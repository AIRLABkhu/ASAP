from collections import deque
import torch as th
from stable_baselines3 import PPO
from typing import Any, ClassVar, Optional, TypeVar, Union
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy

import numpy as np
import torch.nn as nn
from gymnasium import spaces
from torch.nn import functional as F

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



class GRADPPO(PPO):
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
        grad_lamT = 0.1,
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
        self.grad_lamT = grad_lamT


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

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
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

                ### caps start
                weight_t = self.grad_lamT

                observations = rollout_data.observations.clone()
                next_observations = rollout_data.observations.clone().detach()[1:]
                last_obs = rollout_data.observations.clone().detach()[-1].unsqueeze(0)  # 마지막 원소 추가 (배치 차원 유지)
                next_observations = th.cat([next_observations, last_obs], dim=0)  # 마지막 원소 복사하여 추가
                prev_observations = rollout_data.observations.clone().detach()[:-1]
                first_obs = rollout_data.observations.clone().detach()[0].unsqueeze(0)
                prev_observations = th.cat([first_obs, prev_observations], dim=0)

                # 정책 및 가치 함수의 출력 계산
                pi_s = self.policy._predict(observations, deterministic=True).type(th.float32)
                pi_s_next = self.policy._predict(next_observations, deterministic=True).type(th.float32)
                pi_s_prev = self.policy._predict(prev_observations, deterministic=True).type(th.float32)

                # 거리 계산 (여기서는 L2 거리 사용)
                derv_t = 0.5 * ((2*pi_s - pi_s_next - pi_s_prev)**2)

                delta = pi_s_next - pi_s_prev + 1e-4
                hdelta = F.tanh((1/delta)**2).detach()

                loss_t = th.mean(derv_t*hdelta)


                ### caps end

                ### smoothness start
                osc = calculate_oscillation(actions)
                oscillations.append(osc)
                smoothness = calculate_smoothness(actions)
                smoothnesses.append(smoothness)

                ### smoothness end

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + loss_t * weight_t

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


#Grad + CAPS_Spatial
class GRAD_CS_PPO(PPO):
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
        grad_lamT = 0.1,
        grad_lamS = 0.1,
        grad_sigma = 0.2
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
        self.grad_lamT = grad_lamT
        self.grad_lamS = grad_lamS
        self.grad_sigma = grad_sigma


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

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
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

                ### caps start
                weight_t = self.grad_lamT
                weight_s = self.grad_lamS

                observations = rollout_data.observations.clone()
                next_observations = rollout_data.observations.clone().detach()[1:]
                last_obs = rollout_data.observations.clone().detach()[-1].unsqueeze(0)  # 마지막 원소 추가 (배치 차원 유지)
                next_observations = th.cat([next_observations, last_obs], dim=0)  # 마지막 원소 복사하여 추가
                prev_observations = rollout_data.observations.clone().detach()[:-1]
                first_obs = rollout_data.observations.clone().detach()[0].unsqueeze(0)
                prev_observations = th.cat([first_obs, prev_observations], dim=0)

                # 정책 및 가치 함수의 출력 계산
                pi_s = self.policy._predict(observations, deterministic=True).type(th.float32)
                pi_s_next = self.policy._predict(next_observations, deterministic=True).type(th.float32)
                pi_s_prev = self.policy._predict(prev_observations, deterministic=True).type(th.float32)

                # 거리 계산 (여기서는 L2 거리 사용)
                derv_t = 0.5 * ((2*pi_s - pi_s_next - pi_s_prev)**2)

                delta = pi_s_next - pi_s_prev + 1e-4
                hdelta = F.tanh((1/delta)**2).detach()

                loss_t = th.mean(derv_t*hdelta)

                # spatial
                s_bar = observations + th.normal(mean=0.0, std=self.grad_sigma, size=observations.size()).to(observations.device)
                pi_s_bar = self.policy._predict(s_bar, deterministic=True).type(th.float32)

                loss_s = 0.5 * th.nn.functional.mse_loss(pi_s, pi_s_bar)


                ### caps end

                ### smoothness start
                osc = calculate_oscillation(actions)
                oscillations.append(osc)
                smoothness = calculate_smoothness(actions)
                smoothnesses.append(smoothness)

                ### smoothness end

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + loss_t * weight_t + loss_s * weight_s

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


#Grad + grad_Spatial
class GRAD_LS_PPO(PPO):
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
        grad_lamT = 0.1,
        grad_sigma = 1.0,
        grad_lamD = 0.01,
        grad_lamU = 1.0,
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
        self.grad_lamT = grad_lamT
        self.grad_sigma = grad_sigma
        self.grad_lamD = grad_lamD 
        self.grad_lamU = grad_lamU


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

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
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

                ### caps start
                weight_t = self.grad_lamT

                observations = rollout_data.observations.clone()
                next_observations = rollout_data.observations.clone().detach()[1:]
                last_obs = rollout_data.observations.clone().detach()[-1].unsqueeze(0)  # 마지막 원소 추가 (배치 차원 유지)
                next_observations = th.cat([next_observations, last_obs], dim=0)  # 마지막 원소 복사하여 추가
                prev_observations = rollout_data.observations.clone().detach()[:-1]
                first_obs = rollout_data.observations.clone().detach()[0].unsqueeze(0)
                prev_observations = th.cat([first_obs, prev_observations], dim=0)

                # 정책 및 가치 함수의 출력 계산
                pi_s = self.policy._predict(observations, deterministic=True).type(th.float32)
                pi_s_next = self.policy._predict(next_observations, deterministic=True).type(th.float32)
                pi_s_prev = self.policy._predict(prev_observations, deterministic=True).type(th.float32)

                # 거리 계산 (여기서는 L2 거리 사용)
                derv_t = 0.5 * ((2*pi_s - pi_s_next - pi_s_prev)**2)

                delta = pi_s_next - pi_s_prev + 1e-4
                hdelta = F.tanh((1/delta)**2).detach()

                loss_t = th.mean(derv_t*hdelta)

                # spatial
                observations = rollout_data.observations.clone()
                next_observations = rollout_data.observations.clone().detach()[1:]
                last_obs = rollout_data.observations.clone().detach()[-1].unsqueeze(0)  # 마지막 원소 추가 (배치 차원 유지)
                next_observations = th.cat([next_observations, last_obs], dim=0)  # 마지막 원소 복사하여 추가
                # u ~ Uniform(0, 1) 샘플링
                sigma = self.grad_sigma
                ulam = self.grad_lamU
                dlam = self.grad_lamD
                epsilon = sigma * dlam / (ulam - sigma * dlam)
                width = sigma + (sigma - 1) * epsilon
                u = th.rand_like(observations) * 2 * width - width
                # s_bar 계산
                s_bar = observations + (next_observations - observations) * u

                # dus 계산
                diff = (s_bar - observations) / (next_observations - observations + 1e-8)  # Avoid division by zero
                d_us = th.norm(diff, p=float("inf"), dim=-1).detach() + epsilon

                # 정책 및 가치 함수의 출력 계산
                num_mc = 4
                hellinger_terms = []

                #policy distance
                for _ in range(num_mc):
                    pi_s_dist = self.policy.get_distribution(observations)
                    pi_s_bar_dist = self.policy.get_distribution(s_bar)
                    now_actions = self.policy._predict(rollout_data.observations, deterministic=False)
                    log_p_s = pi_s_dist.log_prob(now_actions)
                    log_q_sbar = pi_s_bar_dist.log_prob(now_actions)
                    # pi_s_probs = pi_s_dist.log_prob(now_actions).exp().clamp(min=1e-8)
                    # pi_s_bar_probs = pi_s_bar_dist.log_prob(now_actions).exp().clamp(min=1e-8)
                    

                    # 거리 계산 (여기서는 L2 거리 사용)
                    # policy_distance = th.nn.functional.mse_loss(pi_s, pi_s_bar)
                    # value_distance = th.nn.functional.mse_loss(v_s, v_s_bar)
                    # policy_distance = 0.5 * th.square(th.sqrt(pi_s_probs) - th.sqrt(pi_s_bar_probs))
                    sqrt_ratio = th.exp(0.5 * (log_q_sbar - log_p_s))
                    # print(f"sqrt_ratio_dim : {sqrt_ratio.shape}")
                    hellinger_terms.append(sqrt_ratio)
                    
                
                hellinger_terms_tensor = th.stack(hellinger_terms, dim=1)
                mean_sqrt_ratio = hellinger_terms_tensor.mean(dim=1)
                d_pi_per_sample = (1.0 - mean_sqrt_ratio).clamp(min=0.0) 

                # l2c2 정규화 손실
                lambda_pi = epsilon * ulam / d_us  # 정책 정규화 가중치

                l2c2_loss = th.mean(lambda_pi * d_pi_per_sample)

                ### caps end

                ### smoothness start
                osc = calculate_oscillation(actions)
                oscillations.append(osc)
                smoothness = calculate_smoothness(actions)
                smoothnesses.append(smoothness)

                ### smoothness end

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + loss_t * weight_t + l2c2_loss

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