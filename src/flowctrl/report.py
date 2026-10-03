
import json
from pathlib import Path

def render(results: dict) -> str:
    lines = [
        "# FlowCtrl evaluation",
        "",
        f"Expert demonstrations: {results['dataset']['episodes']} episodes, "
        f"{results['dataset']['samples']} state-action pairs, "
        f"{results['dataset']['expert_success_rate']:.1%} success.",
        "",
        "| policy | condition | success | collision episodes | smoothness | safety margin | upper/lower successes |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for policy, block in results["models"].items():
        for condition in ("nominal", "perturbed"):
            r = block[condition]
            modes = r["successful_mode_counts"]
            lines.append(
                f"| {policy} | {condition} | {r['success_rate']:.1%} | "
                f"{r['collision_episode_rate']:.1%} | {r['mean_action_smoothness']:.4f} | "
                f"{r['mean_min_safety_margin']:.4f} | {modes['upper']}/{modes['lower']} |"
            )
    return "\n".join(lines) + "\n"

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path, default=Path("artifacts/summary.md"))
    args=p.parse_args()
    results=json.loads(args.results.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(results))
    print(render(results))

if __name__=="__main__":
    main()
