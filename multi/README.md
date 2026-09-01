# `multi/` — multi-GPU Boltz-2 inference

Scales the existing searchers across every GPU on the box without editing any of
them. Detects the GPU count at start-up, so the same command is correct on
1×5090, 2×5090 or 4×5090.

## Run it

```bash
python3 -m multi.selftest                  # verify: 4 molecules, real Boltz, ~1 min
python3 -m multi.bench                     # optional: calibrate workers/GPU (box must be idle)

python3 -m multi.run hunter.py --rxn-id 4 --boltz-budget 150 --batch-size 60
python3 -m multi.run orchestrator.py --rxn-id 2
python3 -m multi.run neurons/late_stage_search.py --rxn-id 1
python3 -m multi.run miner/miner.py
python3 -m multi.run neurons/genetic.py
```

`multi.run` executes the target exactly as `python3 <script> <args>` would —
same `__main__`, same argv — with one substitution: the name `boltz_wrapper`
resolves to the multi-GPU facade. **`hunter.py`, `orchestrator.py`,
`late_stage_search.py`, `miner.py` and `genetic.py` are unmodified.**

Under pm2, only the interpreter arguments change:

```bash
pm2 start hunter.py --name hunter-rxn4 \
    --interpreter /root/workspace/nova-4090/.venv/bin/python \
    --interpreter-args "-m multi.run" \
    --cwd /root/workspace/nova-4090 -- --rxn-id 4 --batch-size 60
```

## Raise `--batch-size` when you use this

This is the one setting you must change, and it matters more than the GPU count.

The searchers drive Boltz in batches of `--batch-size` (default **10**), and the
pool can only parallelise what it is given in one call. Ten molecules across
eight workers is two molecules per worker, each paying the full fixed setup.

Setup was measured at **22 s per call** — 12 s of manifest checking and structure
checkpoint load, 10 s of affinity checkpoint load — against **~7.8 s of GPU per
molecule**. Boltz reloads the checkpoint on every `predict()`, so that 22 s is
paid once per chunk no matter how many GPUs are running:

| chunk size | setup overhead |
|---|---|
| 10 | 22% |
| 20 | 12% |
| 30 | 8.6% |
| 48 | 5.6% |

Set `--batch-size` to at least `30 × n_workers` (with 4 GPUs × 2 workers that is
240; a whole round of 150 in one call is better still). This is worth ~15% on a
**single** GPU too, independent of anything in this directory.

## Why a worker pool rather than `predict(devices=N)`

`boltz.main.predict` does accept `devices=N` and drives Lightning's
`DDPStrategy` ([main.py:980-995](../boltz/src/boltz/main.py)). It was not used
because:

- It re-forms a process group on every call and needs a free rendezvous port —
  a searcher calls Boltz every round, for days.
- The 22 s of per-call setup is paid **on every rank, on every call**. Running N
  independent single-GPU wrappers pays it on N different clocks and overlaps one
  worker's setup with another's compute.
- The caller still has to walk the output tree itself, so the input/output
  handling has to be written either way.

The pool also gives dynamic load balancing: workers pull from one shared queue,
so a worker that draws small ligands takes the next chunk instead of idling.
Molecule cost varies 3-4× with heavy-atom count, and any static split leaves a
straggler.

## Why more than one worker per GPU

A single Boltz call does not keep the GPU busy. Sampling `nvidia-smi` at 3 s
through steady-state scoring: **mean utilisation 62.6%**, with 7 of 20 samples
at 0% and VRAM back down to 804 MiB — those are the setup gaps. Peak VRAM per
worker was **5,936 MiB**, so a 32 GiB card has room to spare.

A second worker fills the first one's gaps. `workers_per_gpu = 2` is the default
and the only value with evidence behind it; `multi.bench` measures whether 3 or
4 helps on your box.

## Host RAM is a real limit — and it bit

Placement checks free VRAM **and** `MemAvailable`. That is not defensive
programming: on this box (60 GiB RAM), a `hunter.py` that had been running 14
hours held ~50 GiB RSS, and starting two workers next to it took the machine out
of memory. The kernel killed the searcher mid-round — no traceback, pm2 restarted
it. VRAM was never the constraint; 26 GiB of 32 was free.

`plan_workers()` therefore reserves `NOVA_MULTI_HOST_RESERVE_MIB` (default
12 GiB) for the parent and trims the plan to fit. **Long-running searchers grow.
Check `free -g` before starting a pool next to one.**

## Environment

| variable | effect |
|---|---|
| `NOVA_MULTI_WORKERS_PER_GPU` | override the auto-detected worker count |
| `NOVA_MULTI_GPU_IDS` | restrict to some cards, e.g. `0,2` |
| `NOVA_BOLTZ_GPUS` | explicit worker→GPU map, upstream format: `0,0,1,1` |
| `NOVA_WORKER_VRAM_MIB` | per-worker VRAM budget for placement (default 7000) |
| `NOVA_MULTI_WORKER_RSS_MIB` | per-worker host RAM budget (default 6000) |
| `NOVA_MULTI_HOST_RESERVE_MIB` | RAM left for the parent (default 12000) |
| `NOVA_BOLTZ_THREADS` | CPU threads per worker (default 1) |

`CUDA_VISIBLE_DEVICES` is honoured: the pool only places workers on cards it
leaves visible, so per-reaction GPU pinning still works.

## Files

| file | role |
|---|---|
| `topology.py` | GPU/RAM detection and worker placement |
| `worker.py` | one persistent process, pinned to one GPU, owning a real `BoltzWrapper` |
| `pool.py` | spawns and supervises workers; chunking, retries, restarts |
| `wrapper.py` | `MultiGPUBoltz`, the drop-in for `BoltzWrapper` |
| `patch.py` | installs the drop-in under the name `boltz_wrapper` |
| `run.py` | launcher — runs an unmodified searcher through the patch |
| `selftest.py` | end-to-end correctness check |
| `bench.py` | measures workers/GPU, writes `calibration.json` |

## Correctness

The scoring maths is **not** reimplemented. Each worker runs the real
`BoltzWrapper` and returns what it produced, so heavy-atom normalisation, the
metric combination and the sentinel behaviour stay in the file the validator
also uses. `wrapper.py` reproduces only the bookkeeping of `_postprocess_data`:
which SMILES belongs to which uid, and in what order.

`molecule_scores` is returned in **input SMILES order** (after de-duplication),
which `orchestrator.py:1317` depends on explicitly. `multi.selftest` asserts it.

Spot check against ground truth — `rxn:4:181652:229125`, epoch 24793:

```
validator      0.128199
multi pool     0.129013
```

Inside the ~0.008 spread measured between our scores and the validator's.

## Chunk composition changes scores (this is not new, but it is now visible)

A molecule's Boltz score depends on **which other molecules share its batch**.
Measured here, same molecules, same seed, same GPU:

```
probe-1 in a chunk of 4  ->  0.129013
probe-1 in a chunk of 3  ->  0.093370
```

Re-running with identical chunking reproduces every digit exactly:

```
run A: 0.093370  0.098075  0.108882  0.110202  0.094042  0.101734
run B: 0.093370  0.098075  0.108882  0.110202  0.094042  0.101734
```

So the pool is deterministic given a chunk layout; the variation comes from the
manifest, because `predict()` seeds globally and the per-record sampling stream
depends on batch contents. **This already happens today** — changing
`--batch-size` on the single-GPU path changes the numbers the same way. `multi/`
does not introduce it, but it does mean chunk size is a *scoring* parameter and
not only a throughput one:

- Do not compare scores across different `--batch-size` settings.
- The effect is largest for small ligands, because the score is divided by heavy
  atom count — probe-1 has 21.
- It is part of why our scores and the validator's differ (measured sd 0.0095
  over 519 molecules): the validator batches every miner's molecules for an
  epoch together, a composition we cannot reproduce. No amount of re-drawing at
  a fixed seed removes it.
- Conversely, a re-score that lands in a *different* chunk is closer to a genuinely
  independent draw than a same-seed repeat within the same batch, which is what
  makes the submission-time confirmation gate worth its GPU.
