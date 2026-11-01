"""Fresh output adapters for the frozen strict monitor and independent fold check."""
from pathlib import Path
import shutil,subprocess,sys
from ..paths import ROOT,P8,inside
from .._frozen import load
def prepare_run(output):
    out=inside(output)
    if out.exists():raise FileExistsError("Use a new output directory")
    if out.is_relative_to(ROOT/"tools") or out.is_relative_to(ROOT/"data"):raise ValueError("Use runs/ or another fresh active directory")
    for d in ["scripts","protocol","checkpoints","results","logs"]:(out/d).mkdir(parents=True,exist_ok=True)
    shutil.copytree(P8/"protocol",out/"protocol",dirs_exist_ok=True)
    shutil.copy2(P8/"scripts/frozen_strictcv_source.py",out/"scripts")
    for m in ["EWMA","GARCH-t","HS","QRF"]:shutil.copy2(P8/f"checkpoints/input_{m}.pkl",out/"checkpoints")
    for name in ["02_strict_refit.py","03_s7_refit.py","04_evaluate.py"]:
        text=(P8/"scripts"/name).read_text(encoding="utf-8-sig");needle="A=R/'tools/audit_temp/codex_priority8'"
        if needle not in text:raise RuntimeError("Unknown output-root declaration")
        (out/"scripts"/name).write_text(text.replace(needle,"A=Path("+repr(str(out))+")"),encoding="utf-8")
    return out
def run_strict(output,phase="primary",prepare_only=False):
    out=prepare_run(output)
    if not prepare_only:subprocess.run([sys.executable,"-B",str(out/"scripts/02_strict_refit.py"),"--phase",phase,"--workers","2"],check=True,cwd=ROOT)
    return out
def smoke(output,case="HS:2026Q1:S6:10"):
    out=inside(output)
    if out.exists():raise FileExistsError("Use a fresh smoke-output directory")
    out.mkdir(parents=True);shutil.copy2(P8/"independent_checks/protocol_review/protocol_recovery.json",out/"protocol_recovery.json");m=load("independent_fold");m.O=out;result=m.compare_case(case)
    if result["status"]!="PASS":raise RuntimeError(result)
    return result
