
import numpy as np
from .sim import PlanarPushEnv, PushConfig, rollout_expert

def collect_demonstrations(episodes=160, seed=0, config=None):
    rng = np.random.default_rng(seed)
    env = PlanarPushEnv(seed=seed, config=config or PushConfig())
    states, actions, successes = [], [], 0
    modes = {-1: 0, 1: 0}
    for _ in range(episodes):
        mode = int(rng.choice([-1, 1])); modes[mode] += 1
        env.reset(start_jitter=0.06)
        s, a = rollout_expert(env, mode)
        states.append(s); actions.append(a); successes += int(env.success)
    return np.concatenate(states), np.concatenate(actions), {
        "episodes": episodes,
        "samples": int(sum(len(x) for x in states)),
        "expert_success_rate": successes/episodes,
        "mode_counts": {"lower": modes[-1], "upper": modes[1]},
    }
