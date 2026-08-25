"""How much of the score DB can actually be submitted under the 0.7 rule?"""
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd, sqlite3
import score_store as ss
from config.config_loader import load_config
from utils import get_historical_submissions
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ap = argparse.ArgumentParser()
ap.add_argument("--rxn-id", type=int, default=2)
ap.add_argument("--scan", type=int, default=6000)
a = ap.parse_args()

cfg = load_config(); target = cfg['small_molecule_target'][0]
THR = float(cfg['max_similarity_to_historical'])
K = int(cfg.get('num_molecules', 20))
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

hist = get_historical_submissions(target, "molecules")
hm = [Chem.MolFromSmiles(s) for s in hist["SMILES"]]
hfps = GEN.GetFingerprints([m for m in hm if m is not None], numThreads=8)
print(f"historical: {len(hfps)} | threshold <{THR} | k={K}")

db = ss.score_db_path(a.rxn_id)
rows = sqlite3.connect(db).execute(
    "SELECT molecule_name, smiles, score FROM scored_molecules "
    "WHERE available=TRUE AND smiles IS NOT NULL AND smiles!='' "
    "ORDER BY score DESC LIMIT ?", (a.scan,)).fetchall()
print(f"scanning top {len(rows)} available molecules")

out = []
for i, (name, smi, score) in enumerate(rows):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    sim = max(DataStructs.BulkTanimotoSimilarity(GEN.GetFingerprint(m), hfps))
    out.append((name, smi, float(score), sim))
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(rows)}", flush=True)

d = pd.DataFrame(out, columns=["name", "smiles", "score", "max_hist_sim"])
d.to_csv(os.path.join(os.path.dirname(__file__), "submittable_scan.csv"), index=False)

passing = d[d.max_hist_sim < THR].reset_index(drop=True)
print("\n" + "=" * 70)
print(f"pass rate: {len(passing)}/{len(d)}  ({100*len(passing)/max(len(d),1):.1f}%)")
print(f"naive available top-{K} sum      : {d.head(K).score.sum():.5f}   "
      f"(only {int((d.head(K).max_hist_sim<THR).sum())}/{K} submittable)")
if len(passing) >= K:
    sel = passing.head(K)
    print(f"actually submittable top-{K} sum : {sel.score.sum():.5f}")
    print(f"shortfall vs what you look at   : {sel.score.sum()-d.head(K).score.sum():+.5f}")
    print(f"rank of the {K}th submittable molecule in your list: "
          f"{d.index[d.name==sel.iloc[-1]['name']][0]+1}")
    print(f"\nsubmittable top-{K}:")
    for r in sel.itertuples(index=False):
        print(f"  {r.name:<26}{r.score:9.5f}  sim={r.max_hist_sim:.3f}")
else:
    print(f"ONLY {len(passing)} submittable found in the scanned window "
          f"-- fewer than the {K} required.")
