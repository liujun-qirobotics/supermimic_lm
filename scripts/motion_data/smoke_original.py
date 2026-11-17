import torch

from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import gencoord_env_cfg

c = gencoord_env_cfg()
c.commands["motion"].motion_file = "data/motions/walk1_subject1_research.npz"
c.commands["motion"].sampling_mode = "uniform"
e = create_env(c, 2, "cpu")
cfg = SuperMimicRunnerCfg(
  sampling_mode="uniform",
  buffer_size=16,
  dynamics_hidden_dims=(32,),
  actor_hidden_dims=(32,),
  critic_hidden_dims=(32,),
  wm=WorldModelLearnerCfg(batch_size=2, horizon=2, num_batches=1),
  policy=ActorCriticPolicyLearnerCfg(batch_size=2, horizon=2, num_batches=1),
)
r = SuperMimicRunner(e, cfg, task_id="SuperMimic-GenCoord-G1", device="cpu")
learner = r.world_model_learner
learner._compiled_rollout = learner._rollout
learner._compiled_compute_losses = learner._compute_losses
learner._compiled_represent = learner.representation.represent
for _ in range(4):
  r.step()
print("COLLECT OK")
print("WORLD UPDATE", learner.update([r.replay_buffer.sample(2, 2)]))
print("POLICY UPDATE", r.policy_learner.update())
torch.save(
  {"world": r._wm.bundle(), "actor": r.policy_learner.actor.state_dict()},
  "/tmp/supermimic-original-smoke.pt",
)
loaded = torch.load("/tmp/supermimic-original-smoke.pt", weights_only=True)
r._wm.load_bundle(loaded["world"])
r.policy_learner.actor.load_state_dict(loaded["actor"])
print("SAVE LOAD OK")
e.close()
