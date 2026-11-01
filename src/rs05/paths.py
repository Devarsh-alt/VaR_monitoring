from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
P8 = ROOT / "tools/audit_temp/codex_priority8"
def inside(path):
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT):raise ValueError("Path must remain inside the research project")
    return p
