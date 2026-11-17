# 100STYLE Neutral G1 动作检查

本机输入为 `/home/ub/下载/Neutral_FW.bvh` 和 `Frame_Cuts.csv`。裁剪范围按零基半开区间 `[327, 7445)` 解释，因源文件未明确端点约定，边界存在至多一帧的不确定性。`neutral_fw_trajectory.json` 给出源 BVH 的位移与航向统计，不能证明机器人动力学可行。

完整重映射命令（需本机 GMR 环境）：

```bash
/home/ub/wbc/GMR/.venv/bin/python scripts/motion_data/retarget_100style.py \
  '/home/ub/下载/Neutral_FW.bvh' '/home/ub/下载/Frame_Cuts.csv' \
  data/motions/100style_neutral/neutral_fw_gmr.npz
```

它使用实验性的 [100style_g1_ik.json](100style_g1_ik.json)，不是已验证的标定。根部坐标系、骨长、肢体朝向以及足部接触存在适配误差。GMR 输出是完整 60 Hz 修剪动作，适合初步检查，**不是可直接训练的高质量参考动作**。研究版格式可使用 `convert_gmr.py` 转换，但转换只改变格式，不会修复动作质量。

在保留身体和腿部的基础上，手臂使用方向参考、躯干/骨盆避碰和掌向约束独立拟合：

```bash
/home/ub/wbc/GMR/.venv/bin/python scripts/motion_data/fit_100style_arms.py \
  data/motions/100style_neutral/neutral_fw_gmr.npz '/home/ub/下载/Neutral_FW.bvh' \
  data/motions/100style_neutral/neutral_fw_arms_v4.npz
```

v4 联合拟合七个手臂关节，保持手掌侧向轴对齐躯干侧向轴，选用自然垂臂的掌心朝内约定；不是精确追踪人体手掌旋转。对训练模型的手臂与躯干、骨盆、双侧髋/大腿/小腿碰撞几何施加间距惩罚，另检查橡胶手/腕视觉网格凸包与腿部视觉网格凸包，目标间距 2.5 cm，并将逐帧关节速度限制在 3 rad/s。碰撞惩罚是软约束，生成后必须检查全部帧。生成的 `neutral_fw_arms_v4.arm_fit.json` 记录原始及拟合后的臂段方向误差，不能代替视觉检查。可运行 `audit_100style_arms.py 动作文件 --baseline 原始GMR文件`，逐帧验证所配置碰撞几何的间距和身体腿部保持情况；该检查不覆盖全部视觉网格及双臂相互碰撞。

```bash
uv run --no-sync python scripts/motion_data/convert_gmr.py \
  data/motions/100style_neutral/neutral_fw_arms_v4.npz \
  data/motions/100style_neutral/neutral_fw_arms_v4_research.npz --fps 50 --device cpu

uv run --no-sync python scripts/motion_data/view_reference.py \
  data/motions/100style_neutral/neutral_fw_arms_v4.npz
```

播放器使用与当前训练配置 `gencoord_env_cfg` 相同的 G1 与平地场景，从头循环播放整个有效动作。可用 `--speed 0.5` 减速或 `--start 30` 从第 30 秒开始。画面是运动学参考重放，不是物理仿真下的控制策略表现。应重点检查脚悬空、穿地、腿交叉、摆臂和转弯；在质量确认前不要用于训练。100STYLE 人体局部前向为转换后 -Y，而 G1 局部前向为 +X；重映射器对身体姿态施加了对应的 -90° 局部朝向修正。

BONES-SEED 解压和转换结果及其专用筛选代码已删除，下载目录中的原始归档和元数据保留。

手臂 v6 从 v4 开始对完整 14 关节轨迹做迭代平滑，并对靠近碰撞的位置进行局部投影：

```bash
uv run --no-sync python scripts/motion_data/optimize_arm_trajectory.py \
  data/motions/100style_neutral/neutral_fw_arms_v4.npz \
  data/motions/100style_neutral/neutral_fw_arms_v6.npz
uv run --no-sync python scripts/motion_data/audit_100style_arms.py \
  data/motions/100style_neutral/neutral_fw_arms_v6.npz \
  --baseline data/motions/100style_neutral/neutral_fw_gmr.npz
```

此方法是迭代平滑和逐帧约束投影，不是全局收敛保证。碰撞间距与逐关节加速度需要在输出上独立复查；可用 `view_reference.py` 回放 v6 文件。
