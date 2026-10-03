
import argparse, json
from pathlib import Path
from .data import collect_demonstrations
from .eval import evaluate_policy, NOMINAL, PERTURBED
from .policies import BehaviorCloningPolicy, DiffusionPolicy, FlowMatchingPolicy
from .sim import PushConfig

def run(demos=160, eval_episodes=80, max_iter=100, seed=7):
    cfg=PushConfig(); states,actions,info=collect_demonstrations(demos,seed,cfg)
    policies=[BehaviorCloningPolicy(cfg.max_action,seed,max_iter),
              DiffusionPolicy(cfg.max_action,seed+1,max_iter),
              FlowMatchingPolicy(cfg.max_action,seed+2,max_iter)]
    out={"dataset":info,"models":{}}
    for p in policies:
        p.fit(states,actions)
        out["models"][p.name]={
            "nominal":evaluate_policy(p,eval_episodes,seed+100,NOMINAL),
            "perturbed":evaluate_policy(p,eval_episodes,seed+200,PERTURBED)}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--demos",type=int,default=160); ap.add_argument("--eval-episodes",type=int,default=80)
    ap.add_argument("--max-iter",type=int,default=100); ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--out",type=Path,default=Path("artifacts/results.json")); args=ap.parse_args()
    result=run(args.demos,args.eval_episodes,args.max_iter,args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
