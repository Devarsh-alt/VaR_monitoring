"""Restore verified snapshot bytes into a NEW destination; never overlay live work."""
from pathlib import Path
import argparse,csv,hashlib,shutil,json

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(2**20),b""):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--snapshot",required=True,type=Path);p.add_argument("--dest",required=True,type=Path);p.add_argument("--file",action="append");a=p.parse_args();s=a.snapshot.resolve();d=a.dest.resolve()
 if d.exists():raise ValueError("Destination must be NEW; never overlay or erase existing work")
 if d.is_relative_to(s) or s.is_relative_to(d):raise ValueError("Snapshot and destination must be separate")
 rows=list(csv.DictReader((s/"ORIGINAL_FILE_MANIFEST.csv").open(encoding="utf-8")));chosen=[r for r in rows if not a.file or r["relative_path"] in a.file]
 if a.file and {r["relative_path"] for r in chosen}!=set(a.file):raise ValueError("Requested file absent from manifest")
 d.mkdir(parents=True)
 for r in chosen:
  src=s/"project"/r["relative_path"];dst=(d/r["relative_path"]).resolve()
  if not dst.is_relative_to(d):raise ValueError("Manifest path escapes destination")
  if sha(src)!=r["sha256"]:raise ValueError("Snapshot digest changed")
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  if dst.stat().st_size!=int(r["size"]) or sha(dst)!=r["sha256"]:raise ValueError("Restored digest mismatch")
 print(json.dumps(dict(status="PASS",files=len(chosen),destination=str(d))))
if __name__=="__main__":main()
