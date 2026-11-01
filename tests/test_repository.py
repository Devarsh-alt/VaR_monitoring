"""Numerical/path regressions for the compatibility refactor; unittest only."""
import sys,json,hashlib,unittest,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.dont_write_bytecode=True
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from rs05.data import artifact,catalog
from rs05.features import groups,specifications
from rs05.evaluation import rank_metrics,metrics
from rs05.inference import weights,holm
from rs05.targets import reconstruct
from rs05.paths import ROOT as PACKAGE_ROOT
class RepositoryRegression(unittest.TestCase):
 def test_code_manifest(self):
  import csv
  rows=list(csv.DictReader((ROOT/"results/manifests/repository_code_sha256.csv").open()))
  self.assertGreater(len(rows),15)
  for row in rows:self.assertEqual(hashlib.sha256((ROOT/row["path"]).read_bytes()).hexdigest(),row["sha256"])
 def test_root(self):self.assertEqual(ROOT,PACKAGE_ROOT)
 def test_all_canonical_paths_and_hashes(self):
  for name in catalog():self.assertTrue(artifact(name,verify=True).is_file(),name)
 def test_real_quarter_metrics(self):
  import pickle
  cp=pickle.load(open(ROOT/"tools/audit_temp/codex_priority8/checkpoints/fits/primary/HS/2026Q1_S6.pkl","rb"));g=cp["pred"];m=metrics(g)
  self.assertAlmostEqual(m["auc"],roc_auc_score(g.y,g.p_cal),places=12);self.assertAlmostEqual(m["ap"],average_precision_score(g.y,g.p_cal),places=12);self.assertAlmostEqual(m["brier"],brier_score_loss(g.y,g.p_cal.to_numpy(float)),places=12);self.assertAlmostEqual(m["logloss"],log_loss(g.y,g.p_cal.to_numpy(float)),places=12)
 def test_ties(self):self.assertEqual(rank_metrics([0,1,0,1],[.5]*4),(.5,.5))
 def test_weight_seed(self):
  a=weights(37,20);self.assertTrue(np.array_equal(a,weights(37,20)));self.assertEqual(a.shape,(2000,37));self.assertTrue(np.all(a.sum(1)==37))
 def test_holm_reference(self):
  p=np.array([.01,.04,.003,1.]);self.assertTrue(np.allclose(holm(p),[.03,.08,.012,1.],atol=1e-15))
 def test_source_feature_groups(self):
  self.assertEqual({k:len(v) for k,v in groups().items()},{"P":23,"M":10,"V":8,"S":5,"D":4});self.assertEqual(specifications()["S6"],["P","M","V","S","D"])
 def test_raw_target_equivalence(self):
  raw=pd.read_csv(ROOT/"data/monitoring/regime_shifts/RS03_RS01_regime_panel.csv.gz");raw=raw[raw.ticker.eq("AAPL")].sort_values("forecast_date")
  panel=pd.read_csv(ROOT/"data/monitoring/regime_shifts/RS04b_causal_reliability_panel.csv.gz");panel=panel[panel.ticker.eq("AAPL")&panel.model_name.eq("HS")].sort_values("forecast_date")
  self.assertTrue(raw.forecast_date.reset_index(drop=True).equals(panel.forecast_date.reset_index(drop=True)))
  for h in [5,10,20]:
   out=reconstruct(raw.hs_var,raw.actual_return,h)
   for key,col in [("T1",f"deterioration_{h}d"),("T2",f"coverage_failure_{h}d"),("threshold",f"causal_{h}d_loss_threshold")]:self.assertTrue(np.allclose(out[key],panel[col],equal_nan=True,atol=1e-11))
 def test_active_sources_parse(self):
  for p in (ROOT/"src/rs05").rglob("*.py"):ast.parse(p.read_text())
if __name__=="__main__":unittest.main(verbosity=2)
