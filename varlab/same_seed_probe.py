"""Does re-scoring the SAME molecules in the SAME batch at seed 68 change anything?"""
import os, sys, json, shutil, glob, time
import numpy as np
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "boltz"))
import sqlite3
from config.config_loader import load_config
from boltz_wrapper import BoltzWrapper

cfg = load_config()
sub = {"small_molecule_target": cfg["small_molecule_target"],
       "small_molecule_target_clip_interval": cfg["small_molecule_target_clip_interval"],
       "boltz_mode": cfg.get("boltz_mode","max"),
       "boltz_metric": cfg.get("boltz_metric",["affinity_probability_binary","affinity_pred_value"]),
       "combination_strategy": cfg.get("combination_strategy","heavy_atom_normalization")}

rows = sqlite3.connect(os.path.join(BASE,"score_results_2.sqlite")).execute(
    "SELECT molecule_name, smiles FROM scored_molecules WHERE smiles!='' "
    "ORDER BY score DESC LIMIT 5").fetchall()
mols = [{"name": n, "smiles": s} for n, s in rows]

b = BoltzWrapper()
iso = os.path.join(b.tmp_dir, f"iso_same{os.getpid()}")
b.input_dir = os.path.join(iso, "inputs"); b.output_dir = os.path.join(iso, "outputs")
os.makedirs(b.input_dir, exist_ok=True); os.makedirs(b.output_dir, exist_ok=True)
b.config["override"] = True
b.base_seed = 68                      # identical seed every pass

target = cfg["small_molecule_target"][0]
passes = []
for p in range(3):
    shutil.rmtree(os.path.join(b.output_dir, "boltz_results_inputs"), ignore_errors=True)
    for f in glob.glob(os.path.join(b.input_dir, "*.yaml")): os.remove(f)
    os.makedirs(b.input_dir, exist_ok=True)
    vm = {0: {"smiles": [m["smiles"] for m in mols], "names": [m["name"] for m in mols]}}
    sd = {0: {"target_scores": [[]], "antitarget_scores": [[]], "entropy": None,
              "entropy_boltz": None, "block_submitted": None, "push_time": ""}}
    t0 = time.time()
    b.score_molecules(vm, sd, sub)
    fm = b.final_boltz_scores[0][target]
    passes.append({m["name"]: fm.get(m["smiles"]) for m in mols})
    print(f"pass {p+1} done in {time.time()-t0:.1f}s", flush=True)

print("\n=== SAME seed 68, SAME batch, 3 passes ===")
print(f"{'molecule':<26}{'pass1':>12}{'pass2':>12}{'pass3':>12}{'max diff':>12}")
diffs = []
for m in mols:
    v = [passes[i][m["name"]] for i in range(3)]
    if any(x is None for x in v): continue
    d = max(v) - min(v); diffs.append(d)
    print(f"{m['name']:<26}{v[0]:12.6f}{v[1]:12.6f}{v[2]:12.6f}{d:12.6f}")
print(f"\nmax spread across all molecules: {max(diffs):.8f}")
print("VERDICT:", "IDENTICAL - averaging adds nothing" if max(diffs) < 1e-9
      else f"varies by up to {max(diffs):.6f} - averaging is meaningful")
json.dump(passes, open(os.path.join(BASE,"varlab","same_seed.json"),"w"), indent=2)
shutil.rmtree(iso, ignore_errors=True)
