import json,hashlib
from ..paths import ROOT
def catalog():return json.loads((ROOT/"results/canonical/catalog.json").read_text())
def artifact(name,verify=False):
    entry=catalog()[name];p=ROOT/entry["path"]
    if verify and (p.stat().st_size!=entry["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest()!=entry["sha256"]):raise RuntimeError("Artifact mismatch: "+name)
    return p
