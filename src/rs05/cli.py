import argparse,json,subprocess,sys
from .paths import ROOT,P8
def main():
    p=argparse.ArgumentParser(description="RS05 verified local research interface")
    s=p.add_subparsers(dest="command",required=True);s.add_parser("status");s.add_parser("verify")
    c=s.add_parser("artifact");c.add_argument("name");c.add_argument("--verify",action="store_true")
    c=s.add_parser("run-strict");c.add_argument("--out",required=True);c.add_argument("--phase",choices=["primary","ablation","h5","h20","all"],default="primary");c.add_argument("--prepare-only",action="store_true")
    c=s.add_parser("smoke");c.add_argument("--out",required=True);c.add_argument("--case",default="HS:2026Q1:S6:10")
    a=p.parse_args()
    if a.command=="status":print((P8/"reports/RS05_PRIORITY8_FINAL_STATUS.json").read_text())
    elif a.command=="verify":subprocess.run([sys.executable,"-B",str(P8/"scripts/11_verify_release.py")],cwd=ROOT,check=True)
    elif a.command=="artifact":
        from .data import artifact
        print(artifact(a.name,a.verify))
    elif a.command=="run-strict":
        from .monitoring import run_strict
        print(run_strict(a.out,a.phase,a.prepare_only))
    else:
        from .monitoring import smoke
        print(json.dumps(smoke(a.out,a.case),default=str,indent=2))
if __name__=="__main__":main()
