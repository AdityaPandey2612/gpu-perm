import torch
from per.replay import GPUReplayPER

def test_per_replay_sample_shapes():
    if not torch.cuda.is_available():
        return

    buf = GPUReplayPER(capacity=1024, obs_shape=(4,))
    # fill some transitions
    for i in range(200):
        obs = torch.randn(4)
        next_obs = torch.randn(4)
        buf.add(obs, i % 2, 1.0, next_obs, False)

    batch = buf.sample(batch_size=64, step=0)

    assert batch.obs.shape == (64, 4)
    assert batch.next_obs.shape == (64, 4)
    assert batch.act.shape == (64,)
    assert batch.rew.shape == (64,)
    assert batch.done.shape == (64,)
    assert batch.weights.shape == (64,)
    assert batch.indices.shape == (64,)
    assert batch.indices.min().item() >= 0
    assert batch.indices.max().item() < 1024
    assert batch.weights.min().item() > 0.0
    assert batch.weights.max().item() <= 1.0 + 1e-6

def test_update_priorities_runs():
    if not torch.cuda.is_available():
        return

    buf = GPUReplayPER(capacity=256, obs_shape=(4,))
    for i in range(100):
        buf.add(torch.zeros(4), 0, 0.0, torch.zeros(4), False)

    batch = buf.sample(batch_size=32, step=10)
    td = torch.randn(32, device="cuda")
    buf.update_priorities(batch.indices, td)  # should not crash
