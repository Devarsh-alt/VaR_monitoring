"""Run the packaged RS05 monitor into a fresh output directory."""
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon", type=int, default=10, choices=[5,10,20])
    p.add_argument("--targets", nargs="+", default=["T1","T2"], choices=["T1","T2"])
    p.add_argument("--models", nargs="+", default=None, choices=["HS","EWMA","GARCH-t","QRF"])
    p.add_argument("--algos", nargs="+", default=["logit","xgb"], choices=["logit","xgb"])
    p.add_argument("--specs", nargs="+", default=["S0","S1","S2","S3","S4","S5","S6","S7"])
    p.add_argument("--test-start", default="2019Q1")
    p.add_argument("--test-end", default=None)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--negctrl", action="store_true")
    p.add_argument("--tag", default="curated")
    p.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--output-dir", type=Path, help="Optional fresh output directory instead of runs/<run-name>.")
    a=p.parse_args()
    if Path(a.run_name).name != a.run_name or a.run_name in {".",".."}:
        p.error("--run-name must be a single folder name.")
    output=a.output_dir.resolve() if a.output_dir else ROOT/"runs"/a.run_name
    if output.exists():
        p.error(f"Output directory already exists: {output}. Choose a new name.")
    source=ROOT/"src/experiments/regime_shifts/rs05_walkforward.py"
    spec=importlib.util.spec_from_file_location("rs05_packaged_monitor",source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT_DIR=output
    module.run(a)
    print(f"\nSaved this run to: {output}")

if __name__=="__main__":
    main()
