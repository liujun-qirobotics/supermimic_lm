# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:kevinzakka/mjlab_research.git
cd supermimic
```

## Usage

### Current G1 reference-motion and velocity route

See [参考动作与速度训练说明](参考动作与速度训练说明.md) and
[100STYLE data preparation](scripts/motion_data/README.md). Reference quality,
training convergence, GUI and physical joystick validation are still pending.
The joint-command prototype below is retained as a separate historical path.

### Experimental joint-command training

This separate experimental path trains a single actor using simultaneous
body-frame velocity and two pelvis-relative wrist pose targets, without a
motion file. It uses G1 MuJoCo collection, a learned GenCoord dynamics model,
and short BPTT / critic updates. It is a training prototype, not a validated
locomotion or manipulation policy.

Run a small end-to-end smoke test (defaults to CUDA when available):

```bash
uv run --no-sync python -m supermimic.scripts.train_joint \
  --profile smoke --iterations 3 --num-envs 2 \
  --checkpoint logs/joint/smoke.pt
```

Continue the same profile and environment count from a local checkpoint:

```bash
uv run --no-sync python -m supermimic.scripts.train_joint \
  --profile smoke --iterations 1 --num-envs 2 \
  --resume logs/joint/smoke.pt --checkpoint logs/joint/resumed.pt
```

`--profile dev` uses larger, unvalidated training settings; explicitly choose
`--num-envs` and `--iterations` for the available hardware. Checkpoints contain
models, optimizers and replay, but not the simulator state: resume resets the
environments and cuts the replay boundary. The current wrist target uses link
origins, the foot anchor is a soft reward, and the imagined reward omits some
real-simulator reward and contact terms. Reward/termination fidelity, command
coverage, training convergence, playback and GPU performance remain to be
validated. Only load checkpoints you trust.

### Original motion-tracking training

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
