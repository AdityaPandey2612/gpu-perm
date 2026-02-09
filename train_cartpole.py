import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym

from per.replay import GPUReplayPER


@dataclass
class Config:
    env_id: str = "CartPole-v1"
    seed: int = 0

    total_steps: int = 300_000
    warmup_steps: int = 10_000

    capacity: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99

    lr: float = 1e-3
    train_freq: int = 1
    target_update_freq: int = 1000

    # epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000

    # PER
    alpha: float = 0.6
    beta0: float = 0.4
    beta_anneal_steps: int = 200_000

    device: str = "cuda"


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return end
    t = min(1.0, step / float(duration))
    return start + t * (end - start)


def main():
    cfg = Config()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    assert torch.cuda.is_available(), "CUDA not available"

    env = gym.make(cfg.env_id)
    obs, _ = env.reset(seed=cfg.seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q = QNet(obs_dim, n_actions).to(device)
    q_tgt = QNet(obs_dim, n_actions).to(device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)

    # PER replay on GPU
    replay = GPUReplayPER(
        capacity=cfg.capacity,
        obs_shape=(obs_dim,),
        device=cfg.device,
        alpha=cfg.alpha,
        beta0=cfg.beta0,
    )

    ep_return = 0.0
    ep_len = 0
    episode = 0

    t0 = time.time()
    last_log_t = t0
    last_log_step = 0

    for step in range(1, cfg.total_steps + 1):
        eps = linear_schedule(step, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)

        # action selection (env on CPU, model on GPU)
        if random.random() < eps or step <= cfg.warmup_steps:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                o = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                act = int(torch.argmax(q(o), dim=1).item())

        next_obs, rew, terminated, truncated, _ = env.step(act)
        done = terminated or truncated

        replay.add(obs, act, rew, next_obs, done)

        ep_return += float(rew)
        ep_len += 1
        obs = next_obs

        if done:
            episode += 1
            obs, _ = env.reset()
            # print episodic info
            now = time.time()
            steps_per_sec = (step - last_log_step) / max(1e-9, (now - last_log_t))
            print(
                f"step={step}  episode={episode}  return={ep_return:.1f}  len={ep_len}  eps={eps:.3f}  steps/s={steps_per_sec:.0f}"
            )
            ep_return = 0.0
            ep_len = 0
            last_log_t = now
            last_log_step = step

        # training
        if step >= cfg.warmup_steps and (step % cfg.train_freq == 0):
            batch = replay.sample(
                batch_size=cfg.batch_size,
                step=step,
                beta_steps=cfg.beta_anneal_steps,
            )

            # batch tensors are already CUDA
            obs_b = batch.obs.to(device=device, dtype=torch.float32)
            next_obs_b = batch.next_obs.to(device=device, dtype=torch.float32)
            act_b = batch.act.to(device=device, dtype=torch.int64)
            rew_b = batch.rew.to(device=device, dtype=torch.float32)
            done_b = batch.done.to(device=device, dtype=torch.float32)
            w_b = batch.weights.to(device=device, dtype=torch.float32)

            # Q(s,a)
            q_sa = q(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # standard DQN target (not double-DQN)
                q_next_max = q_tgt(next_obs_b).max(dim=1).values
                target = rew_b + cfg.gamma * (1.0 - done_b) * q_next_max

            td = target - q_sa  # TD error (sign matters)
            loss = (w_b * td.pow(2)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

            # update priorities based on |td|
            replay.update_priorities(batch.indices, td.detach())

        # target network update
        if step % cfg.target_update_freq == 0:
            q_tgt.load_state_dict(q.state_dict())

    env.close()
    dt = time.time() - t0
    print(f"Done. Total steps={cfg.total_steps}, wall_time={dt:.1f}s, avg_steps/s={cfg.total_steps/dt:.0f}")


if __name__ == "__main__":
    main()
