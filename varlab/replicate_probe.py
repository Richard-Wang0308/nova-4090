"""Measure Boltz score reproducibility for the same molecule across replicates."""
import os, sys, time, json, shutil, argparse, sqlite3
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "boltz"))

from config.config_loader import load_config
from boltz_wrapper import BoltzWrapper


def subnet_cfg(config):
    return {
        "small_molecule_target": config["small_molecule_target"],
        "small_molecule_target_clip_interval": config["small_molecule_target_clip_interval"],
        "boltz_mode": config.get("boltz_mode", "max"),
        "boltz_metric": config.get("boltz_metric",
                                   ["affinity_probability_binary", "affinity_pred_value"]),
        "combination_strategy": config.get("combination_strategy", "heavy_atom_normalization"),
    }


def score_once(boltz, config, mols, seed):
    """One independent Boltz pass over `mols` at `seed`. Returns {name: score}."""
    boltz.base_seed = seed
    boltz.config["override"] = True
    for sub in ("boltz_results_inputs",):
        shutil.rmtree(os.path.join(boltz.output_dir, sub), ignore_errors=True)
    os.makedirs(boltz.input_dir, exist_ok=True)
    os.makedirs(boltz.output_dir, exist_ok=True)
    for f in os.listdir(boltz.input_dir):
        if f.endswith(".yaml"):
            os.remove(os.path.join(boltz.input_dir, f))

    vm = {0: {"smiles": [m["smiles"] for m in mols], "names": [m["name"] for m in mols]}}
    sd = {0: {"target_scores": [[]], "antitarget_scores": [[]], "entropy": None,
              "entropy_boltz": None, "block_submitted": None, "push_time": ""}}
    boltz.score_molecules(vm, sd, subnet_cfg(config))
    target = config["small_molecule_target"][0]
    fmap = getattr(boltz, "final_boltz_scores", {}).get(0, {}).get(target, {})
    return {m["name"]: fmap.get(m["smiles"]) for m in mols}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rxn-id", type=int, default=2)
    p.add_argument("--n-mols", type=int, default=5)
    p.add_argument("--replicates", type=int, default=4)
    p.add_argument("--seeds", type=str, default="")
    p.add_argument("--out", type=str, default=os.path.join(BASE, "varlab", "probe_results.json"))
    p.add_argument("--affinity-samples", type=int, default=0,
                   help="override diffusion_samples_affinity (0 = leave config value)")
    p.add_argument("--structure-samples", type=int, default=0,
                   help="override diffusion_samples (0 = leave config value)")
    args = p.parse_args()

    config = load_config()
    db = os.path.join(BASE, f"score_results_{args.rxn_id}.sqlite")
    rows = sqlite3.connect(db).execute(
        "SELECT molecule_name, smiles, score FROM scored_molecules "
        "WHERE smiles IS NOT NULL AND smiles!='' ORDER BY score DESC LIMIT ?",
        (args.n_mols,)).fetchall()
    mols = [{"name": n, "smiles": s, "local": float(sc)} for n, s, sc in rows]
    print(f"probing {len(mols)} molecules x {args.replicates} replicates")
    for m in mols:
        print(f"  {m['name']:<24} stored_local={m['local']:.6f}")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds \
        else [68 + 1000 * i for i in range(args.replicates)]

    boltz = BoltzWrapper()
    # Do not share boltz_tmp_files with a live miner: predict() scores every
    # yaml in the input dir, so concurrent runs pollute each other's batches.
    iso = os.path.join(boltz.tmp_dir, f"iso_probe{os.getpid()}")
    boltz.input_dir = os.path.join(iso, "inputs")
    boltz.output_dir = os.path.join(iso, "outputs")
    os.makedirs(boltz.input_dir, exist_ok=True)
    os.makedirs(boltz.output_dir, exist_ok=True)
    print(f"isolated workspace: {iso}")
    if args.affinity_samples:
        boltz.config["diffusion_samples_affinity"] = args.affinity_samples
        print(f"diffusion_samples_affinity -> {args.affinity_samples}")
    if args.structure_samples:
        boltz.config["diffusion_samples"] = args.structure_samples
        print(f"diffusion_samples -> {args.structure_samples}")
    results = {m["name"]: [] for m in mols}
    for i, seed in enumerate(seeds):
        t0 = time.time()
        got = score_once(boltz, config, mols, seed)
        print(f"[rep {i+1}/{len(seeds)} seed={seed}] {time.time()-t0:.1f}s")
        for n, v in got.items():
            results[n].append(v)
            print(f"    {n:<24} {v}")
        json.dump({"seeds": seeds, "mols": mols, "results": results},
                  open(args.out, "w"), indent=2)

    print("\n=== per-molecule reproducibility ===")
    print(f"{'molecule':<24} {'stored':>9} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'range':>9}")
    for m in mols:
        v = [x for x in results[m["name"]] if x is not None]
        if not v:
            continue
        v = np.array(v, float)
        print(f"{m['name']:<24} {m['local']:9.5f} {v.mean():9.5f} {v.std(ddof=1) if len(v)>1 else 0:9.5f} "
              f"{v.min():9.5f} {v.max():9.5f} {v.max()-v.min():9.5f}")


if __name__ == "__main__":
    main()
