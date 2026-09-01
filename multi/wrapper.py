"""
multi/wrapper.py — MultiGPUBoltz, a drop-in replacement for BoltzWrapper.

Every consumer in this repo calls Boltz through exactly one shape:

    boltz.score_molecules(valid_molecules_by_uid, score_dict, subnet_config)
    scores = score_dict[uid]["molecule_scores"][target_idx]   # input order

hunter.py:826, orchestrator.py:1311, neurons/late_stage_search.py:1094,
miner/miner.py:694, neurons/genetic.py:1827 and rescore.py:187 all do that and
nothing else. So the cheapest safe way to go multi-GPU is to keep that contract
byte for byte and change only what sits behind it:

    from boltz_wrapper import BoltzWrapper      ->   from multi import MultiGPUBoltz
    boltz = BoltzWrapper()                            boltz = MultiGPUBoltz()

Nothing else in a calling script has to change, which is why none of them were
edited.

WHAT IS AND IS NOT REPLICATED
-----------------------------
The scoring maths is NOT reimplemented here. Each worker runs the real
BoltzWrapper and returns what it computed, so heavy-atom normalisation, the
metric combination and the sentinel behaviour stay in one place -- the file the
validator also uses. This module only reproduces the *bookkeeping* of
_postprocess_data: which smiles belong to which uid, and in what order.

ORDERING. `molecule_scores` follows the order of `unique_molecules`, filtered to
the uid -- which for the single-uid payload every caller uses is the input SMILES
order after de-duplication. orchestrator.py depends on that explicitly, and its
own length check falls back to `final_boltz_scores` if it ever breaks.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import Any, Dict, List, Optional

import yaml

from ._log import get_logger
from .pool import BoltzPool

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
log = get_logger("multi.wrapper")


def _get_record_id(rec_id, base_seed):
    """Byte-identical to BoltzWrapper._get_record_id so yaml names match."""
    h = hashlib.sha256(str(rec_id).encode()).digest()
    return (int.from_bytes(h[:8], "little") ^ base_seed) % (2 ** 31 - 1)


class MultiGPUBoltz:
    """BoltzWrapper's interface, served by a pool of single-GPU workers."""

    def __init__(self, workers_per_gpu: Optional[int] = None,
                 plan: Optional[List[int]] = None, logger=None,
                 pool: Optional[BoltzPool] = None):
        self.log = logger or log
        config_path = os.path.join(BASE_DIR, "config", "boltz_config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.base_dir = BASE_DIR
        self.base_seed = 68

        # Present so rescore.isolate_boltz_workspace / clear_boltz_workspace and
        # any operator tooling keep working. The pool's real workspaces live one
        # level down, one per worker; these are this process's own and stay
        # empty. The `iso_` prefix is what clear_boltz_workspace requires before
        # it will delete anything.
        self.tmp_dir = os.path.join(BASE_DIR, "boltz", "boltz_tmp_files")
        root = os.path.join(self.tmp_dir, f"iso_multi{os.getpid()}")
        self.input_dir = os.path.join(root, "inputs")
        self.output_dir = os.path.join(root, "outputs")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.unique_molecules: Dict[str, List] = {}
        self.final_boltz_scores: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.per_molecule_components: Dict[int, Dict[str, Dict[str, dict]]] = {}
        self.subnet_config: Dict[str, Any] = {}

        self.pool = pool or BoltzPool(workers_per_gpu=workers_per_gpu,
                                      plan=plan, logger=self.log)

    # -- BoltzWrapper surface ---------------------------------------------

    def score_molecules(self, valid_molecules_by_uid: dict, score_dict: dict,
                        subnet_config: dict) -> None:
        self.subnet_config = subnet_config
        self._build_unique(valid_molecules_by_uid)
        if not self.unique_molecules:
            self._empty(score_dict, subnet_config)
            return

        molecules = [(ids[0][1], smi) for smi, ids in self.unique_molecules.items()]
        scores = self.pool.score(molecules, subnet_config)
        components = self.pool.last_components
        self._distribute(scores, components)
        self._postprocess(score_dict)

    # The single-target entry point some callers use. Same work here: the
    # wrapper already scores every target in subnet_config.
    score_molecules_target = score_molecules

    def shutdown(self) -> None:
        self.pool.shutdown()

    # -- internals ---------------------------------------------------------

    def _build_unique(self, valid_molecules_by_uid: dict) -> None:
        """{smiles: [(uid, mol_idx), ...]}, in first-seen order.

        Mirrors BoltzWrapper._preprocess_data_for_boltz, including the record id,
        so a molecule keeps the same identity whichever path scored it.
        """
        self.unique_molecules = {}
        for uid, data in (valid_molecules_by_uid or {}).items():
            for smiles in (data or {}).get("smiles", []) or []:
                if not smiles:
                    continue
                if smiles not in self.unique_molecules:
                    self.unique_molecules[smiles] = []
                mol_idx = _get_record_id(smiles, self.base_seed)
                self.unique_molecules[smiles].append((uid, mol_idx))

    def _sentinel(self, subnet_config: dict) -> float:
        return math.inf if subnet_config.get("boltz_mode") == "min" else -math.inf

    def _distribute(self, scores: dict, components: dict) -> None:
        """Fan the pool's per-smiles results back out to every uid that sent it."""
        self.final_boltz_scores = {}
        self.per_molecule_components = {}
        targets = self.subnet_config["small_molecule_target"]
        sentinel = self._sentinel(self.subnet_config)

        for smiles, id_list in self.unique_molecules.items():
            by_target = scores.get(smiles, {})
            comps = components.get(smiles, {})
            for uid, _mol_idx in id_list:
                self.final_boltz_scores.setdefault(uid, {})
                self.per_molecule_components.setdefault(uid, {})
                for target in targets:
                    self.final_boltz_scores[uid].setdefault(target, {})[smiles] = \
                        by_target.get(target, sentinel)
                    if comps.get(target) is not None:
                        self.per_molecule_components[uid].setdefault(smiles, {})[target] = \
                            comps[target]

    def _postprocess(self, score_dict: dict) -> None:
        """Byte-compatible with BoltzWrapper._postprocess_data."""
        targets = self.subnet_config["small_molecule_target"]
        sentinel = self._sentinel(self.subnet_config)
        for uid, data in score_dict.items():
            if uid in self.final_boltz_scores:
                smiles_list = [s for s, ids in self.unique_molecules.items()
                               if any(u == uid for u, _ in ids)]
                data["molecule_scores"] = [
                    [self.final_boltz_scores[uid].get(t, {}).get(s, sentinel)
                     for s in smiles_list]
                    for t in targets
                ]
            else:
                data["molecule_scores"] = [[sentinel] for _ in range(len(targets))]

    def _empty(self, score_dict: dict, subnet_config: dict) -> None:
        sentinel = self._sentinel(subnet_config)
        n = len(subnet_config["small_molecule_target"])
        for _uid, data in score_dict.items():
            data["molecule_scores"] = [[sentinel] for _ in range(n)]
