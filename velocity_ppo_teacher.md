# Train
```bash
cd ~/liujun/supermimic_ws/supermimic

set -o pipefail

env -u PYTHONPATH PYTHONUNBUFFERED=1 uv run --no-sync train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids '[0]' \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.logger tensorboard \
  --agent.run-name g1_ppo_teacher 2>&1 |
  tee -a g1_ppo_teacher.full.log |
  grep --line-buffered -E \
    'Learning iteration [0-9]+/[0-9]+|^[[:space:]]*(Run name|Total steps|Steps per second|Collection time|Learning time|Mean [^:]+|Iteration time|Time elapsed|ETA|(Metrics|Episode_Reward|Episode_Metrics|Episode_Termination|Curriculum)/[^:]+):'
```

# 本地tensorboard查看训练日志
```bash
env -u PYTHONPATH tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006
```

# Play
```bash
CKPT='logs/rsl_rl/g1_velocity/2026-11-17_15-40-25_g1_ppo_teacher/model_5000.pt'
# 可视化速度跟踪 viser改为native 原生窗口动态push测试
env -u PYTHONPATH uv run --no-sync play \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint-file "$CKPT" \
  --num-envs 1 \
  --viewer viser
```

- 原生窗口静态push测试
```bash
CKPT='logs/rsl_rl/g1_velocity/2026-11-17_15-40-25_g1_ppo_teacher/model_9999.pt'

env -u PYTHONPATH uv run --no-sync python - "$CKPT" <<'PY'
import sys
import mjlab.scripts.play as play

ckpt = sys.argv[1]
load_original_cfg = play.load_env_cfg
def load_standing_cfg(*args, **kwargs):
    cfg = load_original_cfg(*args, **kwargs)

    # 每次重置或重新采样后，速度指令仍然是零。
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 1.0
    twist.heading_command = False
    twist.ranges.heading = None
    twist.ranges.lin_vel_x = (0.0, 0.0)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)

    cfg.curriculum.pop("command_vel", None)
    cfg.events.pop("push_robot", None)
    return cfg
play.load_env_cfg = load_standing_cfg
sys.argv = [
    "play", "Mjlab-Velocity-Flat-Unitree-G1",
    "--checkpoint-file", ckpt,
    "--num-envs", "1",
    "--viewer", "native",
]
play.main()
PY
```


# Resume train

- 修改model路径

```bash
CKPT='logs/rsl_rl/g1_velocity/实际运行目录/model_5000.pt'

RESUME_RUN="$(basename "$(dirname "$CKPT")")"
RESUME_FILE="$(basename "$CKPT")"
RESUME_STEM="${RESUME_FILE%.pt}"
RESUME_ITER="${RESUME_STEM#model_}"
EXTRA_ITERS=$((10000 - RESUME_ITER))

set -o pipefail

env -u PYTHONPATH PYTHONUNBUFFERED=1 uv run --no-sync train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids '[0]' \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run "^${RESUME_RUN}$" \
  --agent.load-checkpoint "^${RESUME_STEM}\.pt$" \
  --agent.max-iterations "$EXTRA_ITERS" \
  --agent.save-interval 500 \
  --agent.logger tensorboard \
  --agent.run-name g1_ppo_teacher_resume 2>&1 |
  tee -a g1_ppo_teacher_resume.full.log |
  grep --line-buffered -E \
    'Loading (model )?checkpoint|Learning iteration [0-9]+/[0-9]+|^[[:space:]]*(Run name|Total steps|Steps per second|Collection time|Learning time|Mean [^:]+|Iteration time|Time elapsed|ETA|(Metrics|Episode_Reward|Episode_Metrics|Episode_Termination|Curriculum)/[^:]+):'
```
