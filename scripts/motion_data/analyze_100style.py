"""Analyze 100STYLE root translation and heading within the published trim range."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('bvh', type=Path)
  p.add_argument('cuts', type=Path)
  p.add_argument('--output', type=Path, required=True)
  args=p.parse_args()
  style,movement=args.bvh.stem.rsplit('_',1)
  with args.cuts.open(newline='',encoding='utf-8-sig') as f:
    row=next((r for r in csv.DictReader(f) if r['STYLE_NAME']==style),None)
  if row is None:
    raise ValueError('style missing from frame cuts')
  start,stop=int(row[f'{movement}_START']),int(row[f'{movement}_STOP'])
  lines=args.bvh.read_text().splitlines()
  line=next(i for i,x in enumerate(lines) if x.startswith('Frame Time:'))
  fps=1.0/float(lines[line].split(':',1)[1])
  raw=np.loadtxt(lines[line+1:]); data=raw[start:stop]
  if not 0 <= start < stop <= len(raw) or len(data)<3:
    raise ValueError('invalid frame cuts')
  q=Rotation.from_euler('YXZ',data[:,3:6],degrees=True)
  forward=q.apply(np.tile([0,0,1.],(len(data),1)))
  yaw=np.unwrap(np.arctan2(forward[:,0],forward[:,2]))
  window=round(fps)
  yaw_rate=(yaw[window:]-yaw[:-window])/(window/fps)
  positions=data[:,:3]/100
  speed=np.linalg.norm(positions[window:,[0,2]]-positions[:-window,[0,2]],axis=1)/(window/fps)
  segments=[]
  for begin in range(0,len(data)-window*5+1,window*5):
    end=begin+window*5
    segments.append({'start_s':round(begin/fps,2),'end_s':round(end/fps,2),
                     'net_yaw_deg':round(float(np.degrees(yaw[end-1]-yaw[begin])),1),
                     'horizontal_displacement_m':round(float(np.linalg.norm(positions[end-1,[0,2]]-positions[begin,[0,2]])),2)})
  out={'source':str(args.bvh.resolve()),'source_sha256':hashlib.sha256(args.bvh.read_bytes()).hexdigest(),
       'frame_cut_interpretation':'zero-based [start, stop), endpoint convention not documented',
       'start_frame':start,'stop_frame_exclusive':stop,'fps':fps,'frames':len(data),'duration_s':len(data)/fps,
       'heading_method':'YXZ intrinsic BVH root rotation; projected local +Z on XZ plane',
       'mean_abs_one_second_yaw_rate_rad_s':float(np.mean(abs(yaw_rate))),
       'fraction_one_second_abs_yaw_rate_over_0_3_rad_s':float(np.mean(abs(yaw_rate)>.3)),
       'fraction_one_second_moving_over_0_15_m_s_and_turning_over_0_3_rad_s':float(np.mean((speed>.15)&(abs(yaw_rate)>.3))),
       'five_second_segments':segments,'note':'Kinematic heading only; does not establish stable or suitable G1 turns.'}
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({k:v for k,v in out.items() if k!='five_second_segments'},indent=2))

if __name__=='__main__': main()
