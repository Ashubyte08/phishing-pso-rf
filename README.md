# PSO-RF: Phishing Detection with Binary PSO Feature Selection and Random Forest

Code for the Computer Networks research paper by **Ashita Saxena** (Reg. 2427030165, Batch C1),
Manipal University Jaipur.

Binary particle swarm optimisation (PSO) chooses a small feature subset and a Random Forest
classifies each URL. PSO is re-run inside every outer training fold (nested cross-validation),
so test data never influence feature selection.

## Datasets
| ID | Dataset | Link |
|----|---------|------|
| D1 | UCI Phishing Websites | https://archive.ics.uci.edu/dataset/327/phishing+websites |
| D2 | Mendeley web page phishing dataset (Hannousse & Yahiouche) | https://data.mendeley.com/datasets/c2gw7fy2j4/3 |
| D3 | PhiUSIIL Phishing URL Dataset | https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset |

D1 and D3 download automatically (via `ucimlrepo`). For D2, download `dataset_B_05_2020.csv`
and save it as `data/dataset_phishing.csv`.

## How to run
```
pip install -r requirements.txt
python leakage_test.py        # checks that PSO never sees test rows (~30 s)
python phishing_pso_v2.py --folds 5 --rob_folds 3 --particles 8 --iters 12 --fit_rows 1000 --deploy_runs 1 --out results_v2
```
These are the exact settings used in the paper (random seed 42). The run takes about 30 minutes on a laptop.

## Files
- `phishing_pso_v2.py` – full experiment: data loading, nested CV, PSO, classifiers, ablation, robustness tests, statistics, figures
- `leakage_test.py` – test that confirms there is no feature-selection leakage
- `results/` – result tables (CSV/JSON) reported in the paper
