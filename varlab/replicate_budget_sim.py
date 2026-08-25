"""
How many replicates are worth it? Monte Carlo on the real distributions.

mu      : the 3,715 genuinely submittable scores from varlab/submittable_scan.csv
sigma_i : lognormal fit to the six per-molecule sigmas measured by replicate_probe
          (0.00122 .. 0.01384, median 0.00195) -- heterogeneous on purpose, since
          that heterogeneity is what makes a lucky molecule sneak into the top-20.

Each trial: draw the DB's one-shot score, shortlist top M, re-score K times,
pick 20 by the replicate mean, and report the TRUE sum of what was picked.
"""
import os, sys, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

MEASURED_SIGMAS = np.array([0.00122, 0.00148, 0.00166, 0.00223, 0.00273, 0.01384])


def run(mu, sigma, M, K, k_final, rng, trials, use_single_draw=False):
    """Return mean true-sum of the k_final molecules selected under (M, K)."""
    n = len(mu)
    out = np.empty(trials)
    for t in range(trials):
        x = mu + rng.normal(0, sigma)                    # the DB's single draw
        if K == 0:                                       # current behaviour
            pick = np.argpartition(-x, k_final)[:k_final]
            out[t] = mu[pick].sum()
            continue
        short = np.argpartition(-x, M)[:M]                # shortlist on that draw
        m = mu[short] + rng.normal(0, sigma[short] / np.sqrt(K))
        if use_single_draw:
            # folding the original draw back in re-imports the very selection
            # bias we are trying to remove: x is why the molecule was shortlisted
            m = (x[short] + K * m) / (K + 1)
        best = np.argpartition(-m, k_final)[:k_final]
        out[t] = mu[short][best].sum()
    return out.mean(), out.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--budget", type=int, default=600, help="M*K predictions")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    scan = pd.read_csv(os.path.join(HERE, "submittable_scan.csv"))
    mu = scan.loc[scan.max_hist_sim < 0.7, "score"].to_numpy(float)
    mu = np.sort(mu)[::-1]
    rng = np.random.default_rng(a.seed)

    lg = np.log(MEASURED_SIGMAS)
    sigma = np.exp(rng.normal(lg.mean(), lg.std(ddof=1), size=len(mu)))

    print(f"pool of submittable molecules : {len(mu)}")
    print(f"mu range                      : {mu.min():.5f} .. {mu.max():.5f}")
    print(f"simulated sigma median        : {np.median(sigma):.5f} "
          f"(p10 {np.quantile(sigma,.1):.5f}, p90 {np.quantile(sigma,.9):.5f})")
    print(f"trials                        : {a.trials}\n")

    oracle = mu[:a.k].sum()
    base, base_sd = run(mu, sigma, 0, 0, a.k, rng, a.trials)
    print(f"{'':<26}{'true sum':>10}{'vs 1-shot':>11}{'% of oracle gap':>17}")
    print("-" * 66)
    print(f"{'oracle (knows mu)':<26}{oracle:10.4f}{oracle-base:+11.4f}{'100.0%':>17}")
    print(f"{'current: 1 draw, no rescore':<26}{base:10.4f}{0.0:+11.4f}{'0.0%':>17}")

    print(f"\nfixed shortlist M=120, varying K:")
    print("-" * 66)
    for K in (1, 2, 3, 4, 5, 6, 8):
        m, sd = run(mu, sigma, 120, K, a.k, rng, a.trials)
        pct = 100 * (m - base) / (oracle - base)
        print(f"  K={K}  ({120*K:>4} preds){'':<8}{m:10.4f}{m-base:+11.4f}{pct:16.1f}%")

    print(f"\nSAME GPU BUDGET ({a.budget} predictions), traded between M and K:")
    print("-" * 66)
    for K in (1, 2, 3, 4, 5, 6):
        M = a.budget // K
        if M < a.k * 2:
            continue
        m, sd = run(mu, sigma, M, K, a.k, rng, a.trials)
        pct = 100 * (m - base) / (oracle - base)
        print(f"  M={M:>4} x K={K}{'':<10}{m:10.4f}{m-base:+11.4f}{pct:16.1f}%")

    print(f"\nfolding the original draw into the mean (M=120):")
    print("-" * 66)
    for K in (3, 5):
        a_, _ = run(mu, sigma, 120, K, a.k, rng, a.trials, use_single_draw=False)
        b_, _ = run(mu, sigma, 120, K, a.k, rng, a.trials, use_single_draw=True)
        print(f"  K={K}: replicates only {a_:.4f}   |   + original draw {b_:.4f}"
              f"   ({b_-a_:+.4f})")


if __name__ == "__main__":
    main()
