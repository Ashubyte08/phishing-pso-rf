"""Checks that PSO and the mutual-information filter never see outer test rows.
Run:  py leakage_test.py   (takes ~30 s)"""
import numpy as np, importlib.util, types
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
spec = importlib.util.spec_from_file_location("p", "phishing_pso_v2.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
X, y = make_classification(600, 8, random_state=0); X = np.c_[X, np.arange(600)]  # last column = row ID
seen, mi_seen = [], []
orig_run, orig_mi = p.BinaryPSO.run, p.mutual_info_classif
p.BinaryPSO.run = lambda self, Xf, yf: (seen.append(set(Xf[:, -1].astype(int))), orig_run(self, Xf, yf))[1]
p.mutual_info_classif = lambda Xm, ym, **k: (mi_seen.append(set(Xm[:, -1].astype(int))), orig_mi(Xm, ym, **k))[1]
cfg = types.SimpleNamespace(particles=3, iters=2, alpha=0.99, fit_rows=200, seed=1, n_jobs=1)
for f, (tr, te) in enumerate(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y)):
    seen.clear(); mi_seen.clear()
    p.fold_job("T", f, tr, te, X, y, cfg, "full")
    assert all(s.isdisjoint(te) for s in seen + mi_seen), "LEAK: test rows reached feature selection"
    print(f"fold {f}: {len(seen)} PSO runs and {len(mi_seen)} MI call used training rows only - OK")
print("PASSED: no feature-selection leakage")
