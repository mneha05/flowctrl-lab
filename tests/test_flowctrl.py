
import numpy as np
from flowctrl.data import collect_demonstrations
from flowctrl.policies import BehaviorCloningPolicy, DiffusionPolicy, FlowMatchingPolicy
from flowctrl.sim import PlanarPushEnv, rollout_expert

def test_expert_completes_both_obstacle_modes():
    for mode in (-1, 1):
        env = PlanarPushEnv(seed=3)
        env.reset(start_jitter=0.0)
        rollout_expert(env, mode, action_noise=0.0)
        assert env.success
        assert env.collisions == 0
        ys = np.asarray(env.puck_path)[:, 1]
        assert (ys.max() > 0.4) if mode == 1 else (ys.min() < -0.4)

def test_dataset_contains_both_expert_modes_and_successful_demos():
    states, actions, info = collect_demonstrations(24, seed=9)
    assert states.shape[1] == 10
    assert actions.shape[1] == 2
    assert info["expert_success_rate"] >= 0.95
    assert info["mode_counts"]["upper"] > 0
    assert info["mode_counts"]["lower"] > 0

def test_all_policy_families_fit_and_emit_finite_actions():
    states, actions, _ = collect_demonstrations(20, seed=4)
    rng = np.random.default_rng(4)
    policies = [
        BehaviorCloningPolicy(0.12, seed=1, max_iter=15),
        DiffusionPolicy(0.12, seed=2, max_iter=15, steps=6),
        FlowMatchingPolicy(0.12, seed=3, max_iter=15, steps=6),
    ]
    for policy in policies:
        policy.fit(states, actions)
        action = policy.act(states[0], rng)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))
        assert np.linalg.norm(action) <= 0.17
