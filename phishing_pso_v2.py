"""
Leakage-free nested-CV evaluation of binary-PSO feature selection for phishing detection (v2).
Ashita Saxena - Computer Networks research paper.

WHAT CHANGED FROM v1 (see AUDIT in the paper notes):
  * v1 chose ONE feature mask with PSO on rows that were later inside the CV test folds
    (feature-selection leakage). v2 runs PSO separately inside every OUTER training fold;
    the outer test fold never influences PSO, the MI filter, sampling or model fitting.
  * Statistics: fold scores overlap, so plain Wilcoxon on 10 folds is replaced by the
    Nadeau-Bengio corrected resampled t-test, plus McNemar's test on pooled out-of-fold
    predictions, with Holm correction for multiple comparisons. Classifier ranks are
    compared with a Friedman test whose blocks are DATASETS (N=3), reported with caveats.
  * Robustness experiments also run PSO inside each (corrupted / reduced) training fold.

Run (VS Code terminal, Windows):
    py -m pip install numpy pandas scipy scikit-learn matplotlib ucimlrepo joblib
    py phishing_pso_v2.py --smoke          # 3-5 min sanity check -> results_smoke/  (NOT for paper)
    py phishing_pso_v2.py                  # full run              -> results_v2/
Data: ./data/dataset_phishing.csv (Mendeley); UCI 327 and 967 download automatically.
"""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")   # inherited by parallel workers
import argparse, json, platform, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
warnings.filterwarnings("ignore")

DATASETS = ["D1_UCI", "D2_Mendeley", "D3_PhiUSIIL"]
LOCAL = {"D1_UCI": "data/uci_phishing_websites.csv", "D2_Mendeley": "data/dataset_phishing.csv",
         "D3_PhiUSIIL": "data/PhiUSIIL_Phishing_URL_Dataset.csv"}

# =============================================================== data
def load_dataset(name, synthetic=False):
    """Returns X, y (1 = phishing), feature names, and an audit dict. Raises if data missing."""
    if synthetic:   # pipeline test only
        from sklearn.datasets import make_classification
        n, d = {"D1_UCI": (11055, 30), "D2_Mendeley": (11430, 81), "D3_PhiUSIIL": (40000, 50)}[name]
        X, y = make_classification(n, d, n_informative=max(5, d // 6), n_redundant=d // 4,
                                   flip_y=0.03, weights=[0.55], random_state=1)
        return X, y, [f"f{i}" for i in range(d)], {"Dataset": name, "note": "SYNTHETIC"}
    if os.path.exists(LOCAL[name]):
        df = pd.read_csv(LOCAL[name])
    elif name == "D2_Mendeley":
        raise FileNotFoundError("Missing ./data/dataset_phishing.csv (Mendeley dataset_B_05_2020.csv renamed)")
    else:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=327 if name == "D1_UCI" else 967)
        df = pd.concat([d.data.features, d.data.targets], axis=1)
    cols = {c.lower(): c for c in df.columns}
    if name == "D1_UCI":        # UCI 327: Result -1 = phishing, 1 = legitimate
        t = cols["result"]; y = (df[t].astype(int) == -1).astype(int)
    elif name == "D2_Mendeley":  # status: phishing / legitimate
        t = cols["status"]; y = (df[t].astype(str).str.lower() == "phishing").astype(int)
    else:                        # UCI 967 PhiUSIIL: label 1 = legitimate, 0 = phishing
        t = cols["label"]; y = (df[t].astype(int) == 0).astype(int)
    if y.nunique() != 2:
        raise ValueError(f"{name}: target {t} does not have two classes")
    raw = df.drop(columns=[t])
    num = raw.select_dtypes(include=[np.number])
    text_cols = [c for c in raw.columns if c not in num.columns]
    const_cols = [c for c in num.columns if num[c].nunique(dropna=False) <= 1]
    X = num.drop(columns=const_cols)
    audit = {"Dataset": name, "RawRows": len(df), "RawColumns": df.shape[1], "Target": t,
             "TextColumnsDropped": text_cols, "ConstantColumnsDropped": const_cols,
             "MissingValues": int(X.isna().sum().sum()), "FeaturesUsed": X.shape[1],
             "Phishing": int(y.sum()), "Legitimate": int((1 - y).sum()),
             "DuplicateFeatureRows": int(X.duplicated().sum()),
             "DuplicateRowsWithConflictingLabels": int(pd.concat([X, y.rename("_y")], axis=1)
                                                       .drop_duplicates().duplicated(subset=list(X.columns)).sum())}
    # missing values are filled with 0; this is a fixed rule (no statistics learned from data)
    return X.fillna(0).values.astype(float), y.values, list(X.columns), audit

def strat_sample(X, y, n, seed):
    """Stratified subsample of at most n rows (uses only the rows passed in)."""
    if len(y) <= n:
        return X, y
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return X[idx], y[idx]

# =============================================================== models
def make_clf(name, seed):
    return {"LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "NB": lambda: GaussianNB(),
            "KNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
            "DT": lambda: DecisionTreeClassifier(random_state=seed),
            "RF": lambda: RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed),
            "ET": lambda: ExtraTreesClassifier(n_estimators=100, n_jobs=1, random_state=seed),
            "GB": lambda: HistGradientBoostingClassifier(random_state=seed)}[name]()
CLFS = ["LR", "NB", "KNN", "DT", "RF", "ET", "GB"]
# Scalers live inside pipelines, so they are fitted on training folds only.

# =============================================================== binary PSO
class BinaryPSO:
    """Binary PSO (Kennedy & Eberhart 1997), sigmoid transfer, velocity clipping,
    linearly decreasing inertia (Shi & Eberhart 1998).
    Fitness F = alpha * (1 - mean inner-CV accuracy) + (1 - alpha) * k / D   (minimised).
    It only ever sees the data passed to run(), i.e. part of ONE outer training fold."""
    def __init__(self, P=10, T=15, alpha=0.99, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0,
                 vmax=4.0, linear_inertia=True, inner_folds=3, inner_trees=30, seed=0):
        self.P, self.T, self.alpha, self.w_max, self.w_min = P, T, alpha, w_max, w_min
        self.c1, self.c2, self.vmax, self.linear = c1, c2, vmax, linear_inertia
        self.inner_folds, self.inner_trees, self.seed = inner_folds, inner_trees, seed
        self.cache, self.n_evals = {}, 0

    def fitness(self, mask, X, y):
        key = mask.tobytes()
        if key in self.cache:
            return self.cache[key]
        if not mask.any():                       # empty subset -> worst fitness
            f = 1.0
        else:
            rf = RandomForestClassifier(n_estimators=self.inner_trees, n_jobs=1, random_state=self.seed)
            cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)
            acc = cross_val_score(rf, X[:, mask], y, cv=cv, scoring="accuracy").mean()
            f = self.alpha * (1 - acc) + (1 - self.alpha) * mask.sum() / mask.size
            self.n_evals += 1
        self.cache[key] = f
        return f

    def run(self, X, y):
        rng = np.random.default_rng(self.seed)
        D = X.shape[1]
        x = rng.random((self.P, D)) < 0.5
        v = rng.uniform(-1, 1, (self.P, D))
        fit = np.array([self.fitness(p, X, y) for p in x])
        pbest, pfit = x.copy(), fit.copy()
        g = int(pfit.argmin()); gbest, gfit = pbest[g].copy(), pfit[g]
        self.curve = [gfit]
        for t in range(self.T):
            w = self.w_max - (self.w_max - self.w_min) * t / self.T if self.linear else 0.7
            r1, r2 = rng.random((self.P, D)), rng.random((self.P, D))
            xi = x.astype(float)
            v = w * v + self.c1 * r1 * (pbest - xi) + self.c2 * r2 * (gbest - xi)
            v = np.clip(v, -self.vmax, self.vmax)
            x = rng.random((self.P, D)) < 1.0 / (1.0 + np.exp(-v))
            fit = np.array([self.fitness(p, X, y) for p in x])
            better = fit < pfit
            pbest[better], pfit[better] = x[better], fit[better]
            if pfit.min() < gfit:
                g = int(pfit.argmin()); gbest, gfit = pbest[g].copy(), pfit[g]
            self.curve.append(gfit)
        self.gbest, self.gfit = gbest, gfit
        return gbest

def run_pso(Xtr, ytr, cfg, seed, alpha=None, linear=True):
    """PSO on a stratified subsample of the OUTER TRAINING data only."""
    Xf, yf = strat_sample(Xtr, ytr, cfg.fit_rows, seed)
    t0 = time.perf_counter()
    pso = BinaryPSO(cfg.particles, cfg.iters, cfg.alpha if alpha is None else alpha,
                    linear_inertia=linear, seed=seed)
    mask = pso.run(Xf, yf)
    return mask, pso.curve, time.perf_counter() - t0, pso.n_evals, pso.gfit

# =============================================================== evaluation
def metrics(y, prob):
    p = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y, p), Precision=precision_score(y, p, zero_division=0),
                Recall=recall_score(y, p, zero_division=0), F1=f1_score(y, p, zero_division=0),
                AUC=roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
                FPR=fp / max(fp + tn, 1), FNR=fn / max(fn + tp, 1), TP=tp, TN=tn, FP=fp, FN=fn)

def fit_eval(name, Xtr, ytr, Xte, yte, seed):
    m = make_clf(name, seed)
    t0 = time.perf_counter(); m.fit(Xtr, ytr); tt = time.perf_counter() - t0
    t0 = time.perf_counter(); prob = m.predict_proba(Xte)[:, 1]
    pt = (time.perf_counter() - t0) / len(yte) * 1e6
    r = metrics(yte, prob); r.update(TrainTime_s=tt, PredTime_us=pt)
    return r, prob

def fold_job(ds, fold, tr, te, X, y, cfg, mode="full", noise=0.0, train_frac=1.0):
    """One OUTER fold. Everything except the final scoring uses X[tr], y[tr] only."""
    seed = cfg.seed * 1000 + fold
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = X[tr], y[tr].copy(), X[te], y[te]
    if train_frac < 1:                                    # reduce TRAINING data only
        keep, _ = train_test_split(np.arange(len(ytr)), train_size=train_frac, stratify=ytr, random_state=seed)
        Xtr, ytr = Xtr[keep], ytr[keep]
    if noise > 0:                                         # corrupt TRAINING labels only (before PSO)
        flip = rng.choice(len(ytr), int(round(noise * len(ytr))), replace=False)
        ytr[flip] = 1 - ytr[flip]
    D = X.shape[1]; allm = np.ones(D, bool)
    mask, curve, pso_t, n_ev, gfit = run_pso(Xtr, ytr, cfg, seed)
    out = dict(ds=ds, fold=fold, te=te, mask=mask, curve=curve, pso_time=pso_t, n_evals=n_ev,
               gfit=gfit, rows=[], preds={}, abl=[])
    clfs = CLFS if mode == "full" else ["RF"]
    for fs, m in [("All", allm), ("PSO", mask)]:
        for c in clfs:
            r, prob = fit_eval(c, Xtr[:, m], ytr, Xte[:, m], yte, seed)
            r.update(Dataset=ds, fold=fold, Features=fs, Classifier=c, k=int(m.sum()))
            out["rows"].append(r); out["preds"][f"{fs}-{c}"] = prob
    if mode == "full":                                    # ablation, all inside the outer training fold
        k = int(mask.sum())
        rnd = np.zeros(D, bool); rnd[rng.choice(D, k, replace=False)] = True
        Xm, ym = strat_sample(Xtr, ytr, cfg.fit_rows, seed)        # same rows PSO used
        mi = mutual_info_classif(Xm, ym, random_state=seed)
        mim = np.zeros(D, bool); mim[np.argsort(mi)[-k:]] = True
        m4 = run_pso(Xtr, ytr, cfg, seed, alpha=1.0)[0]
        m5 = run_pso(Xtr, ytr, cfg, seed, linear=False)[0]
        for v, m in [("A1 All features (no FS)", allm), ("A2 Random k features", rnd),
                     ("A3 Mutual-information top-k", mim), ("A4 PSO without size penalty (alpha=1)", m4),
                     ("A5 PSO with constant inertia (w=0.7)", m5), ("A6 Proposed PSO-RF", mask)]:
            r, prob = fit_eval("RF", Xtr[:, m], ytr, Xte[:, m], yte, seed)
            r.update(Dataset=ds, fold=fold, Variant=v, k=int(m.sum()))
            out["abl"].append(r); out["preds"][v] = prob
    return out

def nested_cv(ds, X, y, cfg, folds, mode="full", noise=0.0, train_frac=1.0):
    skf = StratifiedKFold(folds, shuffle=True, random_state=cfg.seed)
    jobs = [delayed(fold_job)(ds, f, tr, te, X, y, cfg, mode, noise, train_frac)
            for f, (tr, te) in enumerate(skf.split(X, y))]
    return Parallel(n_jobs=cfg.n_jobs, verbose=0)(jobs)

# =============================================================== statistics
def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold scores."""
    d = np.asarray(a) - np.asarray(b); k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0, d.mean(), (d.mean(), d.mean())
    se = np.sqrt((1 / k + n_test / n_train) * var)
    t = d.mean() / se; p = 2 * stats.t.sf(abs(t), k - 1)
    h = stats.t.ppf(0.975, k - 1) * se
    return t, p, d.mean(), (d.mean() - h, d.mean() + h)

def mcnemar(y, p_a, p_b):
    """Exact McNemar test on pooled out-of-fold predictions (each URL predicted once)."""
    ca, cb = (p_a >= 0.5) == y, (p_b >= 0.5) == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = 1.0 if b + c == 0 else stats.binomtest(b, b + c, 0.5).pvalue
    return b, c, p

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (m - i) * p[j])); adj[j] = run
    return adj

# =============================================================== output helpers
def to_latex(df, path, caption, label):
    """LaTeX table. \\botrule is the Springer sn-jnl rule command (defined by sn-jnl.cls)."""
    cols = list(df.columns)
    L = ["\\begin{table}[h]", "\\centering", f"\\caption{{{caption}}}\\label{{{label}}}", "\\footnotesize",
         "\\begin{tabular}{@{}" + "l" * len(cols) + "@{}}", "\\toprule",
         " & ".join(str(c).replace("_", "\\_").replace("%", "\\%") for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        L.append(" & ".join(str(x).replace("±", "$\\pm$").replace("_", "\\_").replace("%", "\\%") for x in r) + " \\\\")
    L += ["\\botrule", "\\end{tabular}", "\\end{table}"]
    open(path, "w", encoding="utf-8").write("\n".join(L))

def ms(s, pct=True):
    f = 100 if pct else 1
    return f"{s.mean()*f:.2f} ± {s.std(ddof=1)*f:.2f}"

def save(df, out, name, caption):
    df.to_csv(f"{out}/{name}.csv", index=False)
    to_latex(df, f"{out}/{name}.tex", caption, f"tab:{name}")

def savefig(out, name):
    plt.tight_layout(); plt.savefig(f"{out}/{name}.png", dpi=300); plt.savefig(f"{out}/{name}.pdf"); plt.close()

# =============================================================== main
def main(cfg):
    out = cfg.out; os.makedirs(out, exist_ok=True)
    print(f"Output -> {out}/   (n_jobs={cfg.n_jobs})")
    audits, single_feat, all_rows, all_abl, folds_info, deploy = [], [], [], [], [], {}
    oof, pso_curves, cond_rows, stat_rows, mc_rows = {}, {}, [], [], []
    for ds in DATASETS:
        t_ds = time.time()
        print(f"\n===== {ds} =====  [{time.strftime('%H:%M:%S')}]")
        X, y, feats, audit = load_dataset(ds, cfg.synthetic)
        Xe, ye = strat_sample(X, y, cfg.max_rows, cfg.seed)        # evaluation sample (no model decisions)
        audit.update(RowsUsed=len(ye), PhishingUsed=int(ye.sum()), LegitimateUsed=int((1 - ye).sum()),
                     DuplicateRowsInSample=int(pd.DataFrame(Xe).duplicated().sum()))
        audits.append(audit)
        # --- data sanity check: how well does ONE feature separate the classes? (5-fold CV, depth-2 tree)
        sf = []
        for j, fn in enumerate(feats):
            acc = cross_val_score(DecisionTreeClassifier(max_depth=2, random_state=0), Xe[:, [j]], ye,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
            sf.append((fn, acc))
        for fn, acc in sorted(sf, key=lambda z: -z[1])[:5]:
            single_feat.append(dict(Dataset=ds, Feature=fn, SingleFeatureCVAccuracy=round(acc * 100, 2)))
        # --- main nested CV (PSO inside every outer training fold)
        res = nested_cv(ds, Xe, ye, cfg, cfg.folds, "full")
        n_tr = len(ye) * (cfg.folds - 1) / cfg.folds; n_te = len(ye) / cfg.folds
        probs = {}
        for r in res:
            all_rows += r["rows"]; all_abl += r["abl"]
            folds_info.append(dict(Dataset=ds, fold=r["fold"], k=int(r["mask"].sum()), D=len(feats),
                                   PSOTime_s=round(r["pso_time"], 1), FitnessEvals=r["n_evals"],
                                   BestFitness=round(float(r["gfit"]), 4),
                                   Selected=";".join(f for f, b in zip(feats, r["mask"]) if b)))
            for key, p in r["preds"].items():
                probs.setdefault(key, np.zeros(len(ye)))[r["te"]] = p
        oof[ds] = (ye, probs); pso_curves[ds] = np.array([r["curve"] for r in res])
        print(f"  k per fold: {[int(r['mask'].sum()) for r in res]}")
        # --- statistics within this dataset
        df = pd.DataFrame(all_rows); df = df[df.Dataset == ds]
        da = pd.DataFrame(all_abl); da = da[da.Dataset == ds]
        ref = df[(df.Features == "PSO") & (df.Classifier == "RF")].sort_values("fold").F1.values
        comps = [(f"{fs}-{c}", df[(df.Features == fs) & (df.Classifier == c)].sort_values("fold").F1.values)
                 for fs in ["All", "PSO"] for c in CLFS if not (fs == "PSO" and c == "RF")]
        comps += [(v, da[da.Variant == v].sort_values("fold").F1.values)
                  for v in dict.fromkeys(da.Variant) if not v.startswith("A6") and not v.startswith("A1")]
        tt = [corrected_ttest(ref, s, n_tr, n_te) for _, s in comps]
        mcs = [mcnemar(ye, probs["PSO-RF"], probs[n]) for n, _ in comps]
        adj_t, adj_m = holm([t[1] for t in tt]), holm([m[2] for m in mcs])
        for (n, _), t, pt, m, pm in zip(comps, tt, adj_t, mcs, adj_m):
            stat_rows.append(dict(Dataset=ds, Comparison=f"PSO-RF vs {n}", MeanDiffF1_pp=round(t[2] * 100, 3),
                                  CI95_pp=f"[{t[3][0]*100:.2f}, {t[3][1]*100:.2f}]", t=round(t[0], 3),
                                  p_corrected_t=round(t[1], 4), p_Holm_t=round(pt, 4),
                                  McNemar_b=m[0], McNemar_c=m[1], p_McNemar=round(m[2], 4), p_Holm_McNemar=round(pm, 4)))
        # --- deployment subset: PSO on the whole development sample (NOT used for any test estimate)
        best = min((run_pso(Xe, ye, cfg, cfg.seed + s) for s in range(cfg.deploy_runs)), key=lambda z: z[4])
        deploy[ds] = [f for f, b in zip(feats, best[0]) if b]
        # --- robustness (nested CV again; PSO re-run on each modified training fold)
        if not cfg.skip_robustness:
            conds = [("Label noise", lv, dict(noise=lv)) for lv in [0.0, 0.05, 0.10, 0.20]]
            conds += [("Training fraction", lv, dict(train_frac=lv)) for lv in [0.10, 0.25, 0.50]]
            for cname, lv, kw in conds:
                rr = nested_cv(ds, Xe, ye, cfg, cfg.rob_folds, "min", **kw)
                for r in rr:
                    for row in r["rows"]:
                        cond_rows.append(dict(Dataset=ds, Condition=cname, Level=lv, Features=row["Features"],
                                              fold=row["fold"], F1=row["F1"], Recall=row["Recall"], k=row["k"]))
            ph, lg = np.where(ye == 1)[0], np.where(ye == 0)[0]
            rng = np.random.default_rng(cfg.seed)
            for ratio in [0.10, 0.25]:                   # phishing share of the whole sample
                n_ph = min(len(ph), int(ratio / (1 - ratio) * len(lg)))
                idx = np.concatenate([lg, rng.choice(ph, n_ph, replace=False)])
                rr = nested_cv(ds, Xe[idx], ye[idx], cfg, cfg.rob_folds, "min")
                for r in rr:
                    for row in r["rows"]:
                        cond_rows.append(dict(Dataset=ds, Condition="Phishing share", Level=ratio, Features=row["Features"],
                                              fold=row["fold"], F1=row["F1"], Recall=row["Recall"], k=row["k"]))
        print(f"  done in {(time.time()-t_ds)/60:.1f} min")

    # =========================================================== tables
    R = pd.DataFrame(all_rows); A = pd.DataFrame(all_abl); Fi = pd.DataFrame(folds_info)
    R.to_csv(f"{out}/perfold_main.csv", index=False); A.to_csv(f"{out}/perfold_ablation.csv", index=False)
    Fi.to_csv(f"{out}/perfold_pso.csv", index=False)
    json.dump(audits, open(f"{out}/dataset_audit.json", "w"), indent=2, default=str)
    save(pd.DataFrame([{k: a[k] for k in ["Dataset", "RawRows", "FeaturesUsed", "Phishing", "Legitimate",
                                           "RowsUsed", "DuplicateFeatureRows", "DuplicateRowsInSample"] if k in a}
                       for a in audits]), out, "table_datasets", "Datasets and rows used")
    save(pd.DataFrame(single_feat), out, "table_single_feature_check",
         "Top single-feature 5-fold CV accuracy (depth-2 tree), a data sanity check")
    pso_t = Fi.groupby("Dataset").agg(D=("D", "first"), k_mean=("k", "mean"), k_sd=("k", "std"),
                                      k_min=("k", "min"), k_max=("k", "max"), time=("PSOTime_s", "mean")).reset_index()
    pso_t["Reduction_%"] = (100 * (1 - pso_t.k_mean / pso_t.D)).round(1)
    pso_t["k (mean ± sd)"] = pso_t.apply(lambda r: f"{r.k_mean:.1f} ± {r.k_sd:.1f}", axis=1)
    pso_t["k range"] = pso_t.apply(lambda r: f"{int(r.k_min)}-{int(r.k_max)}", axis=1)
    save(pso_t[["Dataset", "D", "k (mean ± sd)", "k range", "Reduction_%", "time"]].rename(columns={"time": "PSO time per fold (s)"}),
         out, "table_pso_selection", "Features selected by PSO inside each outer training fold")
    main_t = R.groupby(["Dataset", "Features", "Classifier"]).agg(
        k=("k", "mean"), Accuracy=("Accuracy", ms), Precision=("Precision", ms), Recall=("Recall", ms),
        F1=("F1", ms), AUC=("AUC", ms), FPR=("FPR", ms), FNR=("FNR", ms),
        TrainTime_s=("TrainTime_s", lambda s: f"{s.mean():.3f}"), PredTime_us=("PredTime_us", lambda s: f"{s.mean():.2f}")).reset_index()
    main_t["k"] = main_t.k.round(1)
    save(main_t, out, "table_main_results", "Nested-CV results (mean ± sd over outer folds, %)")
    abl_t = A.groupby(["Dataset", "Variant"]).agg(k=("k", "mean"), F1=("F1", ms), Recall=("Recall", ms),
                                                  AUC=("AUC", ms)).reset_index()
    abl_t["k"] = abl_t.k.round(1)
    save(abl_t, out, "table_ablation", "Ablation with Random Forest (nested CV; k = mean features)")
    S = pd.DataFrame(stat_rows)
    save(S, out, "table_stats", "PSO-RF vs other configurations: corrected resampled t-test and McNemar test, Holm-adjusted")
    # Friedman over DATASETS (blocks = 3 datasets, treatments = 7 classifiers on PSO features)
    piv = R[R.Features == "PSO"].groupby(["Dataset", "Classifier"]).F1.mean().unstack()[CLFS]
    ranks = piv.rank(axis=1, ascending=False)
    fr = stats.friedmanchisquare(*[piv[c].values for c in CLFS])
    fr_t = pd.DataFrame({"Classifier": CLFS, "MeanRank": ranks.mean().round(2).values,
                         **{f"Rank {d}": ranks.loc[d].values for d in piv.index}}).sort_values("MeanRank")
    save(fr_t, out, "table_friedman_ranks",
         f"Classifier ranks per dataset (Friedman over N=3 datasets: chi2={fr.statistic:.2f}, p={fr.pvalue:.3g})")
    json.dump(dict(chi2=fr.statistic, p=fr.pvalue, N_datasets=3, k=7,
                   note="Blocks are datasets; with N=3 the test has low power and no post-hoc test is applied."),
              open(f"{out}/friedman.json", "w"), indent=2)
    json.dump(deploy, open(f"{out}/deployment_feature_subsets.json", "w"), indent=2)
    freq = []
    for ds in DATASETS:
        sub = Fi[Fi.Dataset == ds]
        cnt = pd.Series(";".join(sub.Selected).split(";")).value_counts()
        for f, c in cnt.items():
            if f: freq.append(dict(Dataset=ds, Feature=f, FoldsSelected=int(c), OfFolds=len(sub)))
    pd.DataFrame(freq).to_csv(f"{out}/feature_selection_frequency.csv", index=False)
    if cond_rows:
        C = pd.DataFrame(cond_rows); C.to_csv(f"{out}/perfold_conditions.csv", index=False)
        ct = C.groupby(["Dataset", "Condition", "Level", "Features"]).agg(F1=("F1", ms), k=("k", "mean")).reset_index()
        ct["k"] = ct.k.round(1)
        save(ct, out, "table_conditions", f"Robustness (nested {cfg.rob_folds}-fold CV, PSO re-run per fold)")

    # =========================================================== figures
    plt.figure(figsize=(6, 4))
    for ds in DATASETS:
        c = pso_curves[ds]; m, s = c.mean(0), c.std(0); it = np.arange(len(m))
        plt.plot(it, m, label=ds); plt.fill_between(it, m - s, m + s, alpha=0.2)
    plt.xlabel("Iteration"); plt.ylabel("Best fitness (lower is better)"); plt.legend()
    plt.title("PSO convergence (mean ± sd over outer folds)"); savefig(out, "fig_convergence")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for i, ds in enumerate(DATASETS):
        g = R[R.Dataset == ds].groupby(["Classifier", "Features"]).F1.mean().unstack().loc[CLFS] * 100
        x = np.arange(len(CLFS))
        ax[i].bar(x - 0.2, g["All"], 0.4, label="All features"); ax[i].bar(x + 0.2, g["PSO"], 0.4, label="PSO (per fold)")
        ax[i].set_xticks(x); ax[i].set_xticklabels(CLFS); ax[i].set_title(ds); ax[i].set_ylabel("F1 (%)")
        ax[i].set_ylim(max(0, g.values.min() - 3), 100.5)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(f"{out}/fig_classifier_f1.pdf"); plt.savefig(f"{out}/fig_classifier_f1.png", dpi=300); plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for i, ds in enumerate(DATASETS):
        yt, P = oof[ds]
        for key, lab in [("All-RF", "RF, all features"), ("PSO-RF", "PSO-RF")]:
            fpr, tpr, _ = roc_curve(yt, P[key]); ax[i].plot(fpr, tpr, label=f"{lab} (AUC={roc_auc_score(yt, P[key]):.4f})")
        ax[i].plot([0, 1], [0, 1], "k--", lw=0.8); ax[i].set_title(ds); ax[i].set_xlabel("FPR"); ax[i].set_ylabel("TPR")
        ax[i].legend(loc="lower right", fontsize=8)
    savefig(out, "fig_roc")
    g = A.groupby(["Variant", "Dataset"]).F1.mean().unstack()[DATASETS] * 100
    fig, ax = plt.subplots(figsize=(9, 4.2)); yy = np.arange(len(g))
    for j, ds in enumerate(DATASETS): ax.barh(yy + (j - 1) * 0.26, g[ds], 0.26, label=ds)
    ax.set_yticks(yy); ax.set_yticklabels(g.index, fontsize=8.5); ax.set_xlabel("F1 (%)")
    ax.set_xlim(max(0, g.values.min() - 3), 100.5); ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    savefig(out, "fig_ablation")
    if cond_rows:
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        for i, cn in enumerate(["Label noise", "Training fraction", "Phishing share"]):
            for j, ds in enumerate(DATASETS):
                for fs, ls in [("All", "--"), ("PSO", "-")]:
                    q = C[(C.Condition == cn) & (C.Dataset == ds) & (C.Features == fs)].groupby("Level").F1.mean() * 100
                    ax[i].plot(q.index, q.values, marker="o", ls=ls, color=f"C{j}", label=f"{ds} ({fs})")
            ax[i].set_title(cn); ax[i].set_xlabel("Level"); ax[i].set_ylabel("F1 (%)")
        h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8)
        plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(f"{out}/fig_conditions.pdf"); plt.savefig(f"{out}/fig_conditions.png", dpi=300); plt.close()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for j, ds in enumerate(DATASETS):
        ax.hist(Fi[Fi.Dataset == ds].k, bins=range(0, 90, 2), alpha=0.6, label=ds)
    ax.set_xlabel("Features selected in an outer fold"); ax.set_ylabel("Folds"); ax.legend(); savefig(out, "fig_k_per_fold")

    cfgd = vars(cfg).copy(); cfgd.update(python=sys.version.split()[0], sklearn=sklearn.__version__,
                                         platform=platform.platform(), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                                         design="nested CV: PSO inside each outer training fold")
    json.dump(cfgd, open(f"{out}/run_config.json", "w"), indent=2)
    print(f"\nDone. Results in ./{out}/  -> zip this folder and send it back.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--particles", type=int, default=10)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--folds", type=int, default=10, help="outer folds, main experiment")
    ap.add_argument("--rob_folds", type=int, default=5, help="outer folds, robustness experiments")
    ap.add_argument("--max_rows", type=int, default=10000, help="stratified development sample per dataset")
    ap.add_argument("--fit_rows", type=int, default=2000, help="PSO sample, drawn from the OUTER TRAINING fold")
    ap.add_argument("--deploy_runs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1, help="parallel outer folds (-1 = all cores)")
    ap.add_argument("--skip_robustness", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny settings -> results_smoke/ (never for the paper)")
    ap.add_argument("--synthetic", action="store_true", help="synthetic data, pipeline test only")
    ap.add_argument("--out", default=None)
    cfg = ap.parse_args()
    if cfg.smoke:
        cfg.particles, cfg.iters, cfg.folds, cfg.rob_folds, cfg.max_rows, cfg.fit_rows, cfg.deploy_runs = 6, 4, 3, 3, 3000, 800, 1
    cfg.out = cfg.out or ("results_smoke" if cfg.smoke else ("results_synthetic" if cfg.synthetic else "results_v2"))
    main(cfg)
