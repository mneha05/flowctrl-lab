# FlowCtrl Lab

[![Generative control verification](https://github.com/mneha05/flowctrl-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/mneha05/flowctrl-lab/actions/workflows/ci.yml)

**Diffusion models, conditional flow matching, and behavior cloning for multimodal robot-control imitation learning.**

FlowCtrl Lab is a compact research-style control project designed around one question:

> When an expert demonstrates multiple valid ways to solve the same manipulation task, how do deterministic behavior cloning and stochastic generative policies behave?

The task is a **simulated planar manipulation environment**: a point end-effector acquires a puck and moves it to a goal while routing around an obstacle. Expert demonstrations deliberately contain two legitimate modes—**above** or **below** the obstacle—so the dataset is multimodal rather than a single averaged trajectory.

This is simulated control research. It does **not** claim physical-robot deployment or real-world sim-to-real transfer.

## Verified experiment

GitHub Actions run [#37084299618](https://github.com/mneha05/flowctrl-lab/actions/runs/37084299618) trained all three policies from **100 expert demonstrations / 2,284 state-action pairs** and evaluated **30 nominal + 30 perturbed closed-loop episodes per policy**.

The expert demonstration set had **100.0% task success** with both upper and lower obstacle routes represented.

| policy | condition | success | collision episodes | action smoothness | mean safety margin | successful upper/lower routes |
|---|---|---:|---:|---:|---:|---:|
| behavior cloning | nominal | **86.7%** | 0.0% | 0.0192 | 0.2749 | 16 / 10 |
| behavior cloning | perturbed | **76.7%** | 0.0% | 0.0192 | 0.2686 | 11 / 12 |
| diffusion | nominal | **60.0%** | 3.3% | 0.0659 | 0.2723 | 14 / 4 |
| diffusion | perturbed | **73.3%** | 10.0% | 0.0634 | 0.2064 | 13 / 9 |
| flow matching | nominal | **73.3%** | 0.0% | 0.0528 | 0.2360 | 8 / 14 |
| flow matching | perturbed | **70.0%** | 6.7% | 0.0512 | 0.2572 | 7 / 14 |

The result is intentionally reported as measured: in this small experiment the deterministic BC baseline has the highest nominal success, while both stochastic generative policies still recover successful trajectories in both demonstration modes. The workflow uploads the full JSON results and Markdown evaluation summary as the `policy-evaluation` artifact.

## Three policies, same demonstrations

### 1. Behavior cloning baseline

A conditional MLP directly regresses the normalized expert action from the robot state with mean-squared error. This is the standard imitation-learning baseline and makes mode averaging visible when expert actions disagree.

### 2. Conditional diffusion policy

The policy trains a DDPM-style denoiser on expert actions. For a clean normalized action `a` and Gaussian noise `eps`, training creates a noisy action at diffusion step `t` and learns to predict the injected noise:

```text
x_t = sqrt(alpha_bar_t) * a + sqrt(1 - alpha_bar_t) * eps
epsilon_theta(state, x_t, t) -> eps
```

At rollout time, each action begins from Gaussian noise and is iteratively denoised while conditioned on the current robot state.

### 3. Conditional flow-matching policy

The flow policy uses a straight-line probability path between Gaussian noise `z` and expert action `a`:

```text
x_t = (1 - t) * z + t * a
v*(x_t, t) = a - z
```

A conditional MLP learns the vector field `v_theta(state, x_t, t)`. Inference starts from Gaussian noise and integrates the learned field with Euler steps to generate an action.

## Imitation-learning dataset

The expert controller is hand-coded only to generate demonstrations. It:

1. approaches and acquires the puck;
2. randomly commits to the upper or lower obstacle route;
3. moves the puck around the obstacle with collision-free waypoints;
4. finishes at the goal.

Training consumes **state-action demonstration pairs only**. The learned policies do not receive the expert route label.

State includes end-effector position, puck position, goal, obstacle geometry, and normalized time. Action is a 2-D end-effector displacement.

## Policy evaluation

Every trained policy is evaluated by full closed-loop rollouts rather than offline action error alone.

Metrics:

- **task success rate**
- **collision-episode rate**
- **final goal distance**
- **action smoothness**: mean `||a_t - a_(t-1)||`
- **puck path length**
- **minimum obstacle safety margin**
- **successful upper/lower route counts** as a simple multimodality check

Two environments are evaluated:

**Nominal:** same dynamics family used for demonstrations.

**Perturbed (sim-to-real-style stress test):**

- observation noise;
- randomized actuator gain;
- randomized contact efficiency;
- shifted obstacle position.

These perturbations are intentionally described as **sim-to-real-style robustness tests**, not physical sim-to-real validation.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
pytest -q

flowctrl-experiment \
  --demos 160 \
  --eval-episodes 80 \
  --max-iter 100 \
  --out artifacts/results.json

flowctrl-report artifacts/results.json
```

## What the repo demonstrates

- score/noise-prediction training for a conditional diffusion control policy;
- conditional flow matching and ODE-style action generation;
- behavior cloning from expert demonstrations;
- multimodal expert behavior;
- closed-loop manipulation-policy evaluation;
- success, safety, smoothness, and robustness metrics;
- controlled dynamics/observation perturbations;
- honest comparison against a deterministic BC baseline.

The purpose is **not** to force the generative policies to “win.” CI records the measured results for all three methods. A useful research result can include a strong BC baseline, failure modes, or sensitivity to sampling/training choices.

## Scope boundary

This is a small simulated 2-D manipulation study, not a claim of physical robot manipulation experience. It closes the implementation gap around diffusion/flow control, behavior cloning, demonstration learning, and policy evaluation while keeping physical-robot and production-control claims separate.

## License

MIT
