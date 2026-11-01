"""Load existing verified numerical code without copying/changing its implementation."""
import hashlib,importlib.util,json,sys
from functools import lru_cache
from .paths import ROOT
@lru_cache(None)
def load(name):
    registry=json.loads((ROOT/"configs/frozen_protocols/source_registry.json").read_text())
    entry=registry[name];path=ROOT/entry["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:raise RuntimeError("Frozen source digest changed: "+str(path))
    spec=importlib.util.spec_from_file_location("rs05_frozen_"+name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module
