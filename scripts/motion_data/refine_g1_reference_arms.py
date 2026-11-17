"""Fit reference G1 arm segment directions while preserving its body and legs."""
import argparse
from pathlib import Path
import mujoco,numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from arm_ik_100style import ArmDirectionIK

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 with np.load(a.source,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
 model=mujoco.MjModel.from_xml_path(str(G1_XML));solver=ArmDirectionIK(model,fps=float(data['fps']));d=solver.data;q=data['qpos'];names=[model.joint(i).name for i in range(1,model.njnt)]
 if names!=data['joint_names'].tolist():raise ValueError('G1 joint ordering mismatch')
 out=[]
 for state in q:
  d.qpos[:]=state;mujoco.mj_kinematics(model,d)
  targets={side:np.diff(d.xpos[solver.sides[side]['body']],axis=0).copy() for side in ('left','right')}
  out.append(solver.fit(state,targets))
 data['qpos']=np.stack(out);data['robot_xml']=np.asarray(str(G1_XML));data['arm_fit']='g1_reference_direction_collision_neutral_palm';a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**data)
 print(a.source.name,'frames',len(out),'saved',a.output,flush=True)
if __name__=='__main__':main()
