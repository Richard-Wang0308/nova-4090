"""
main.py — dispatcher for the 4 mining modes.

Usage:
    python main.py --mode 1 --rxn_id 2      # DPEX_DJA  (continuous loop)
    python main.py --mode 2 --rxn_id 2      # Exhaust   (continuous loop)
    python main.py --mode 3 --rxn_id 2      # Input     (runs once, merges CSV, exits)
    python main.py --mode 3 --rxn_id 2 --input_csv /path/to/new_rxn2.csv
    python main.py --mode 4 --rxn_id 2      # Cross     (continuous loop)

Each mode can also be run directly, e.g.:
    python mode2_exhaust.py --rxn_id 2
"""

import argparse
import asyncio

import common
from common import logger
from config.config_loader import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Nova SN68 Miner — 4-mode dispatcher")
    parser.add_argument(
        "--mode", type=int, required=True, choices=[1, 2, 3, 4],
        help="1=DPEX_DJA  2=Exhaust  3=Input  4=Cross",
    )
    parser.add_argument("--rxn_id", type=int, required=True, help="Reaction ID (e.g. 1-5)")
    parser.add_argument(
        "--input_csv", type=str, default=None,
        help="Mode 3 only: path to external score CSV. Defaults to data/rxn{ID}.csv",
    )
    return parser.parse_args()


async def run_async_mode(mode: int, rxn_id: int):
    common.configure_for_rxn(rxn_id)
    config = load_config()
    common.initialize_solution(config, rxn_id)
    state = common.build_state(config)

    if mode == 1:
        import mode1_dpex_dja as m
        await m.find_solution(state, rxn_id)
    elif mode == 2:
        import mode2_exhaust as m
        await m.run_exhaust_loop(state, rxn_id)
    elif mode == 4:
        import mode4_cross as m
        await m.run_cross_loop(state, rxn_id)
    else:
        raise ValueError(f"Unsupported async mode: {mode}")


def run_sync_mode(mode: int, rxn_id: int, input_csv: str):
    common.configure_for_rxn(rxn_id)

    if mode == 3:
        import mode3_input as m
        import os
        csv_path = input_csv or os.path.join(common.BASE_DIR, "data", f"rxn{rxn_id}.csv")
        m.run_input_merge(rxn_id, csv_path)
    else:
        raise ValueError(f"Unsupported sync mode: {mode}")


def main():
    args = parse_args()

    mode_names = {1: "DPEX_DJA", 2: "Exhaust", 3: "Input", 4: "Cross"}
    logger.info(f"🚀 Starting Mode {args.mode} ({mode_names[args.mode]}) | rxn={args.rxn_id}")

    if args.mode == 3:
        # Input mode is synchronous, runs once, exits
        run_sync_mode(args.mode, args.rxn_id, args.input_csv)
        logger.info("✅ Input mode complete — exiting")
        return

    try:
        asyncio.run(run_async_mode(args.mode, args.rxn_id))
    except KeyboardInterrupt:
        logger.info(f"✅ Mode {args.mode} stopped by user (rxn={args.rxn_id})")


if __name__ == "__main__":
    main()
