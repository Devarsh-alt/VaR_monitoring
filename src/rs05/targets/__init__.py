"""Reuse the independent historical reconstruction by exact AST extraction."""
import ast,bisect,hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import binom
from ..paths import ROOT
def reconstruct(q,r,h):
    entry=json.loads((ROOT/"configs/frozen_protocols/source_registry.json").read_text())["targets"];p=ROOT/entry["path"]
    if hashlib.sha256(p.read_bytes()).hexdigest()!=entry["sha256"]:raise RuntimeError("Target reference source changed")
    tree=ast.parse(p.read_text(encoding="utf-8-sig"));fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="reconstruct")
    ns=dict(np=np,pd=pd,bisect=bisect,binom=binom);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),"exec"),ns)
    return ns["reconstruct"](q,r,h)
