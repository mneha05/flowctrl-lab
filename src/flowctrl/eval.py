
from dataclasses import dataclass, asdict
import numpy as np
from .sim import PlanarPushEnv, PushConfig

@dataclass
class EvalCondition:
    name: str
    observation_noise: float = 0.0
    action_scale_low: float = 1.0
    action_scale_high: float = 1.0
    contact_eff_low: float = 0.90
    contact_eff_high: float = 0.90
    obstacle_shift_std: float = 0.0

NOMINAL = EvalCondition("nominal")
PERTURBED = EvalCondition("perturbed", 0.018, 0.82, 1.08, 0.76, 0.94, 0.035)

def evaluate_policy(policy, episodes=80, seed=100, condition=NOMINAL):
    rng=np.random.default_rng(seed); env=PlanarPushEnv(seed=seed,config=PushConfig()); rows=[]
    for ep in range(episodes):
        scale=rng.uniform(condition.action_scale_low,condition.action_scale_high)
        eff=rng.uniform(condition.contact_eff_low,condition.contact_eff_high)
        shift=rng.normal(0,condition.obstacle_shift_std,2)
        env.reset(start_jitter=0.06,obstacle_shift=tuple(shift),contact_efficiency=eff)
        erng=np.random.default_rng(seed*10000+ep); acts=[]
        for _ in range(env.cfg.horizon):
            a=policy.act(env.observe(condition.observation_noise),erng); acts.append(np.asarray(a))
            _,done=env.step(a,action_scale=scale)
            if done: break
        acts=np.asarray(acts); path=np.asarray(env.puck_path)
        smooth=float(np.mean(np.linalg.norm(np.diff(acts,axis=0),axis=1))) if len(acts)>1 else 0
        plen=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()) if len(path)>1 else 0
        mode="upper" if np.max(path[:,1])>=abs(np.min(path[:,1])) else "lower"
        rows.append({"success":env.success,"collisions":env.collisions,"goal_distance":env.goal_distance,
                     "smoothness":smooth,"path_length":plen,"min_safety_margin":float(env.min_safety_margin),"mode":mode})
    success=[r for r in rows if r["success"]]; modes={"upper":0,"lower":0}
    for r in success: modes[r["mode"]]+=1
    return {"condition":asdict(condition),"episodes":episodes,
            "success_rate":float(np.mean([r["success"] for r in rows])),
            "collision_episode_rate":float(np.mean([r["collisions"]>0 for r in rows])),
            "mean_goal_distance":float(np.mean([r["goal_distance"] for r in rows])),
            "mean_action_smoothness":float(np.mean([r["smoothness"] for r in rows])),
            "mean_path_length":float(np.mean([r["path_length"] for r in rows])),
            "mean_min_safety_margin":float(np.mean([r["min_safety_margin"] for r in rows])),
            "successful_mode_counts":modes}
