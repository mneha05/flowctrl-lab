
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class PushConfig:
    horizon: int = 60
    max_action: float = 0.12
    contact_radius: float = 0.20
    contact_efficiency: float = 0.90
    goal_radius: float = 0.15
    obstacle_radius: float = 0.25

class PlanarPushEnv:
    def __init__(self, seed: int = 0, config: PushConfig | None = None):
        self.rng = np.random.default_rng(seed)
        self.cfg = config or PushConfig()
        self.reset()

    def reset(self, start_jitter=0.05, obstacle_shift=(0.0, 0.0), contact_efficiency=None):
        self.ee = np.array([-1.0, 0.0]) + self.rng.normal(0, start_jitter, 2)
        self.puck = np.array([-0.52, 0.0]) + self.rng.normal(0, start_jitter * 0.5, 2)
        self.goal = np.array([1.0, 0.0])
        self.obstacle = np.array([0.18, 0.0]) + np.array(obstacle_shift, dtype=float)
        self.contact_efficiency = self.cfg.contact_efficiency if contact_efficiency is None else float(contact_efficiency)
        self.t = 0
        self.collisions = 0
        self.grasped = False
        self.actions = []
        self.puck_path = [self.puck.copy()]
        self.min_safety_margin = float("inf")
        return self.observe()

    def observe(self, noise_std=0.0):
        state = np.concatenate([self.ee, self.puck, self.goal, self.obstacle, [self.cfg.obstacle_radius, self.t / self.cfg.horizon]])
        if noise_std:
            state = state + self.rng.normal(0, noise_std, state.shape)
        return state.astype(np.float64)

    def _clip(self, action, scale=1.0):
        a = np.asarray(action, dtype=float).reshape(2) * scale
        n = np.linalg.norm(a)
        if n > self.cfg.max_action:
            a = a / n * self.cfg.max_action
        return a

    def step(self, action, action_scale=1.0):
        a = self._clip(action, action_scale)
        old_ee = self.ee.copy()
        new_ee = old_ee + a
        touching = np.linalg.norm(old_ee-self.puck) <= self.cfg.contact_radius or np.linalg.norm(new_ee-self.puck) <= self.cfg.contact_radius
        if touching:
            self.grasped = True
        if self.grasped:
            candidate = self.puck + self.contact_efficiency * a
            if np.linalg.norm(candidate-self.obstacle) <= self.cfg.obstacle_radius:
                self.collisions += 1
            else:
                self.puck = candidate
        self.ee = new_ee
        self.t += 1
        self.actions.append(a.copy())
        self.puck_path.append(self.puck.copy())
        self.min_safety_margin = min(self.min_safety_margin, float(np.linalg.norm(self.puck-self.obstacle)-self.cfg.obstacle_radius))
        done = self.success or self.t >= self.cfg.horizon
        return self.observe(), done

    @property
    def success(self):
        return np.linalg.norm(self.puck-self.goal) <= self.cfg.goal_radius

    @property
    def goal_distance(self):
        return float(np.linalg.norm(self.puck-self.goal))

def _toward(src, dst, max_step):
    d = np.asarray(dst)-np.asarray(src)
    n = np.linalg.norm(d)
    return np.zeros(2) if n < 1e-9 else d/n*min(max_step, n)

def expert_action(env: PlanarPushEnv, mode: int):
    assert mode in (-1, 1)
    cfg = env.cfg
    if not env.grasped:
        return _toward(env.ee, env.puck, cfg.max_action)

    side_y = mode * (cfg.obstacle_radius + 0.34)
    left_x = env.obstacle[0] - cfg.obstacle_radius - 0.12
    right_x = env.obstacle[0] + cfg.obstacle_radius + 0.16

    if env.puck[0] < left_x and abs(env.puck[1]) < abs(side_y) * 0.90:
        target = np.array([left_x, side_y])
    elif env.puck[0] < right_x - 0.03:
        target = np.array([right_x, side_y])
    else:
        target = env.goal
    return _toward(env.puck, target, cfg.max_action)

def rollout_expert(env, mode, action_noise=0.006):
    states, actions = [], []
    for _ in range(env.cfg.horizon):
        s = env.observe()
        a = expert_action(env, mode)
        if action_noise:
            a = a + env.rng.normal(0, action_noise, 2)
        states.append(s); actions.append(a)
        _, done = env.step(a)
        if done: break
    return np.asarray(states), np.asarray(actions)
