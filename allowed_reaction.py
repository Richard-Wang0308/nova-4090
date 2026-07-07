#!/usr/bin/env python3
"""
Print allowed reaction for a given epoch.

Uses the same challenge helper used in miner/validator:
`utils.get_challenge_params_from_blockhash`.
"""

import argparse
import asyncio
import os
import sys

import bittensor as bt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from config.config_loader import load_config
from utils import get_challenge_params_from_blockhash, get_total_reactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get allowed reaction for a specific epoch"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Epoch number (e.g. 22208)",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        default=361,
        help="Epoch length in blocks (default: 361)",
    )
    parser.add_argument(
        "--network",
        default="finney",
        help="Bittensor network (default: SUBTENSOR_NETWORK or finney)",
    )
    parser.add_argument(
        "--rxn-only",
        action="store_true",
        help="Fail if allowed reaction is not rxn:1..rxn:5",
    )
    return parser.parse_args()


def validate_rxn_only(allowed_reaction: str) -> bool:
    if not allowed_reaction or not allowed_reaction.startswith("rxn:"):
        return False
    try:
        rid = int(allowed_reaction.split(":", 1)[1])
    except (IndexError, ValueError):
        return False
    return 1 <= rid <= 5


def resolve_challenge_params(block_hash: str, cfg: dict) -> dict:
    """Resolve challenge params using utils.challenge.get_challenge_params_from_blockhash."""
    small_molecule_target = cfg.get("small_molecule_target", cfg.get("weekly_target", ""))
    if isinstance(small_molecule_target, list):
        small_molecule_target = small_molecule_target[0] if small_molecule_target else ""

    nanobody_target = cfg.get("nanobody_target", small_molecule_target)
    if isinstance(nanobody_target, list):
        nanobody_target = nanobody_target[0] if nanobody_target else ""

    return get_challenge_params_from_blockhash(
        block_hash=block_hash,
        small_molecule_target=small_molecule_target,
        nanobody_target=nanobody_target,
        num_antitargets=cfg.get("num_antitargets") or 0,
        include_reaction=cfg.get("random_valid_reaction", True),
    )


async def main() -> None:
    args = parse_args()
    cfg = load_config()

    epoch_start_block = args.epoch * args.epoch_length

    subtensor = bt.async_subtensor(network=args.network)
    await subtensor.initialize()
    try:
        start_block_hash = await subtensor.determine_block_hash(epoch_start_block)

        total_reaction_count = get_total_reactions()

        challenge_params = resolve_challenge_params(start_block_hash, cfg)

        allowed_reaction = (
            challenge_params.get("allowed_reaction") if challenge_params else None
        )

        print(f"epoch={args.epoch}")
        print(f"start_block={epoch_start_block}")
        print(f"start_block_hash={start_block_hash}")
        print(f"total_reaction_count={total_reaction_count}")
        print(f"final_allowed_reaction={allowed_reaction}")

        if args.rxn_only and not validate_rxn_only(allowed_reaction):
            raise SystemExit(
                f"allowed_reaction is not rxn:1..rxn:5: {allowed_reaction}"
            )
    finally:
        await subtensor.close()


if __name__ == "__main__":
    asyncio.run(main())

