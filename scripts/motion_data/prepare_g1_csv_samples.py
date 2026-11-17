"""Convert a small list of existing BONES-SEED G1 CSVs for complete playback."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mjlab.scripts.csv_to_npz import JOINT_NAMES


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--fps',type=float,default=120.)
 choices={'walk':'walk_ff_loop_180_R','backward':'walk_ff_loop_360_R','turn_right':'turn_walk_270_R','turn_left':'turn_walk_270_R_M','carry_forward':'lift_crate_walk_ff_loop_180_R','carry_side':'lift_crate_walk_ff_loop_270_R','carry_turn':'lift_crate_turn_270'}
 p.add_argument('--only',choices=tuple(choices),default=None)
 p.add_argument('--per-class',type=int,default=2)
 p.add_argument('--exclude-actors',nargs='*',default=[])
 p.add_argument('--actors',nargs='*',default=None)
 # Parse after registering selection options.
 args=p.parse_args()
 if args.per_class < 1: p.error('--per-class must be positive')
 if args.only: choices={args.only:choices[args.only]}
 if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError('output directory must be empty')
 with args.metadata.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 args.output_dir.mkdir(parents=True,exist_ok=True);records=[]
 for label,name in choices.items():
  selected=[r for r in rows if r['content_name']==name and r['is_mirror'].lower() in (('true','1') if label=='turn_left' else ('false','0'))]
  actors=set();unique=[]
  for row in selected:
   if row['take_actor'] not in actors and row['take_actor'] not in args.exclude_actors and (args.actors is None or row['take_actor'] in args.actors):
    actors.add(row['take_actor']);unique.append(row)
   if len(unique)==args.per_class:break
  if len(unique)!=args.per_class:raise ValueError(f'insufficient examples for {name}')
  for row in unique:
   relative=Path(row['move_g1_path']); source=(args.dataset_root/relative).resolve()
   if not source.is_relative_to(args.dataset_root.resolve()):raise ValueError('source path escapes dataset root')
   with source.open(encoding='utf-8-sig',newline='') as f: header=next(csv.reader(f))
   if tuple(header[:7])!=('Frame','root_translateX','root_translateY','root_translateZ','root_rotateX','root_rotateY','root_rotateZ'):raise ValueError('unexpected root columns')
   actual=[n.removesuffix('_dof') for n in header[7:]]
   if len(actual)!=29 or len(set(actual))!=29 or set(actual)!=set(JOINT_NAMES):raise ValueError('G1 joint mismatch')
   data=np.loadtxt(source,delimiter=',',skiprows=1,ndmin=2)
   if data.shape[1]!=36 or len(data)<3 or not np.isfinite(data).all() or not np.allclose(np.diff(data[:,0]),1):raise ValueError('invalid source frames')
   q=Rotation.from_euler('xyz',data[:,4:7],degrees=True).as_quat()[:,[3,0,1,2]]
   pos=np.column_stack((data[:,1:4]/100,q,np.deg2rad(data[:,7:][:,[actual.index(x) for x in JOINT_NAMES]])))
   output=args.output_dir/f'{label}_{row["take_actor"]}.npz'
   np.savez_compressed(output,qpos=pos.astype(np.float32),joint_names=np.array(JOINT_NAMES),fps=args.fps,quaternion_order='wxyz')
   records.append({'label':label,'file':output.name,'source':str(relative),'actor':row['take_actor'],'description':row['content_short_description'],'frames':len(pos),'fps_assumed':args.fps,'duration_s':len(pos)/args.fps,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'status':'requires visual inspection'})
 (args.output_dir/'manifest.json').write_text(json.dumps({'clips':records},indent=2)+'\n')
 for r in records:print(r['file'],r['frames'],f"{r['duration_s']:.2f}s",r['description'])
if __name__=='__main__':main()
