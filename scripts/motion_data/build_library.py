"""Build a reproducible LAFAN1 candidate library; quality review is separate."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256(path):
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--source-dir", type=Path, required=True)
  p.add_argument("--gmr-python", type=Path, required=True)
  p.add_argument("--output-dir", type=Path, required=True)
  p.add_argument("--patterns", nargs="+", default=["walk*.bvh", "aiming*.bvh"])
  args = p.parse_args()
  here = Path(__file__).resolve().parent
  clips = sorted(
    {f.resolve() for pattern in args.patterns for f in args.source_dir.glob(pattern)}
  )
  if not clips:
    p.error("No input motions found")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  intermediate = args.output_dir / "gmr"
  converted = args.output_dir / "research"
  logs = args.output_dir / "logs"
  for directory in (intermediate, converted, logs):
    directory.mkdir(exist_ok=True)
  env = dict(
    os.environ,
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MPLCONFIGDIR="/tmp/matplotlib",
  )
  for i, source in enumerate(clips, 1):
    raw = intermediate / (source.stem + ".npz")
    final = converted / (source.stem + ".npz")
    manifest = converted / (source.stem + ".json")
    print(f"[{i}/{len(clips)}] {source.name}", flush=True)
    if final.exists() or raw.exists() or manifest.exists():
      print(
        f"SKIP existing outputs for {source.name}; inspect before replacing", flush=True
      )
      continue
    with (logs / (source.stem + ".log")).open("w") as log:
      commands = [
        [str(args.gmr_python), str(here / "retarget_lafan1.py"), str(source), str(raw)],
        [sys.executable, str(here / "convert_gmr.py"), str(raw), str(final)],
      ]
      try:
        for command in commands:
          subprocess.run(
            command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT
          )
      except subprocess.CalledProcessError:
        print(f"FAILED {source.name}; see {log.name}", flush=True)
        continue
    record = {
      "source": str(source),
      "source_sha256": sha256(source),
      "gmr_sha256": sha256(raw),
      "motion_sha256": sha256(final),
      "original_fps": 30,
      "output_fps": 50,
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(f"COMPLETE {source.name}", flush=True)


if __name__ == "__main__":
  main()
