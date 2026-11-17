"""Apply a constant waist-pitch offset to a G1 reference, preserving all other joints."""
import argparse
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--offset-deg',type=float,default=-4.0);a=p.parse_args()
 with np.load(a.source,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
 i=7+data['joint_names'].tolist().index('waist_pitch_joint');data['qpos'][:,i]+=np.deg2rad(a.offset_deg);data['waist_pitch_adjustment_deg']=a.offset_deg
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**data)
if __name__=='__main__':main()
