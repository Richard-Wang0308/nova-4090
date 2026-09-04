#!/usr/bin/env python3
"""
LATE-EPOCH VARIANT OF miner/submit.py

The ONLY difference from miner/submit.py is WHEN the submission fires:

    miner/submit.py       submits as soon as an epoch begins -- immediately on
                          startup, then immediately on every epoch change.
    miner/late_submit.py  submits near the END of the epoch: once only
                          LATE_SUBMIT_BLOCKS_REMAINING (31) blocks are left,
                          i.e. after 330 of the epoch's 361 blocks have passed.

Everything else is IMPORTED from miner/submit.py rather than copied -- molecule
selection, the entropy floor and its score-maximising search, the unconfirmed
donor tier, timelock encryption, the concurrent chain commits over one
persistent connection, the batched GitHub upload, and all availability
bookkeeping. A forked copy would drift the moment either file is edited, and
these two must stay identical everywhere except the trigger. Only
run_epoch_loop() is redefined here.

Run it exactly like the normal miner:

    python3 miner/late_submit.py --wallet.name nova --netuid 68 \\
        --network finney --logging.debug

Do NOT run this alongside miner/submit.py for the same wallet/hotkey: both
would submit for the same epoch from the same molecule database, and the
second submission would spend molecules the first one already spent.
"""

import os
import sys
import asyncio
import datetime
import traceback
from typing import Any, Dict

from dotenv import load_dotenv
import bittensor as bt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Importing the miner also installs its SIGINT/SIGTERM handlers and its
# shutdown_event, which this loop honours.
#
# submit.py is a SIBLING of this file, and how it has to be named depends on how
# this file was launched. Run as a script (python3 miner/late_submit.py) the
# script's own directory leads sys.path, so `import submit` finds it -- and
# `from miner import submit` does NOT work there, because this directory also
# holds miner.py, which claims the name `miner` ahead of the package and fails
# with "cannot import name 'submit' from 'miner'". Run as a module
# (python3 -m miner.late_submit) it is the repo root that leads sys.path, and
# the package form is the one that resolves. Both are supported.
try:
    import submit as base                 # python3 miner/late_submit.py
except ImportError:                       # pragma: no cover
    from miner import submit as base      # python3 -m miner.late_submit


# ============================================================================
# LATE-TRIGGER CONFIGURATION
# ============================================================================

# Blocks still to run in the epoch when the submission fires. At the 361-block
# epoch and ~12 s per block this is ~6 minutes before the boundary, and it is
# the same instant as "330 blocks into the epoch".
LATE_SUBMIT_BLOCKS_REMAINING = 41

# The trigger is expressed in BLOCKS REMAINING, not as a hardcoded block 330,
# and is measured against the epoch length the loop reads from the chain
# (state['epoch_length']). If that ever differs from EPOCH_LENGTH = 361 -- a
# runtime value, not a constant -- a hardcoded 330 would fire at the wrong
# distance from the boundary, early or late depending on which way it moved.
# At 361 the two are the same thing: 361 - 31 = 330.

# Refuse to START a submission with fewer than this many blocks left.
#
# The whole pipeline has to finish INSIDE the epoch it is submitting for. A
# chain commit that lands after the boundary is read against the next epoch,
# whose allowed reaction and target protein are different, so it is discarded
# -- and mark_molecules_unavailable() still burns all twenty molecules on
# commit success. Skipping costs the epoch; overrunning costs the epoch AND the
# molecules, so when there is clearly not enough time left, skipping wins.
#
# Measured end-to-end cost of do_epoch_submission()'s selection stage on this
# box: 0.5 s when the score-ordered top set already clears the entropy floor,
# ~45 s when the floor-constrained search has to run, and ~122 s on the donor
# tier (the slowest path: a failed confirmed-only attempt, then loading and
# qualifying 4000 donor candidates, then a second search). 122 s is ~10 blocks;
# 13 blocks (~2.6 min) covers it with room for the commit and upload.
#
# In steady state this guard never fires: the loop triggers with 31 blocks left
# and polls every POLL_INTERVAL seconds. It exists for the startup case, when
# the process happens to come up inside the last few blocks of an epoch.
LATE_SUBMIT_MIN_BLOCKS =20


def _trigger_block(epoch_length: int) -> int:
    """Blocks into the epoch at which the submission fires."""
    return max(0, epoch_length - LATE_SUBMIT_BLOCKS_REMAINING)


# ============================================================================
# MAIN EPOCH LOOP -- the one thing this file changes
# ============================================================================

async def run_epoch_loop(state: Dict[str, Any]) -> None:
    """
    Monitoring and submission loop, firing LATE in the epoch.

    Workflow:
    1. Poll the chain every POLL_INTERVAL seconds.
    2. Submit for the current epoch the first time that epoch is seen with
        LATE_SUBMIT_BLOCKS_REMAINING or fewer blocks to go.
    3. Act at most once per epoch, tracked by last_acted_epoch.
    4. Repeat.

    Unlike miner/submit.py this does NOT submit on startup. Starting mid-epoch
    before the trigger means waiting for it; starting inside the window submits
    straight away, unless too little of the epoch is left to finish safely (see
    LATE_SUBMIT_MIN_BLOCKS).
    """
    bt.logging.info("🔄 Starting epoch monitoring loop (LATE-SUBMIT mode)...\n")

    last_acted_epoch = None
    last_status_log = datetime.datetime.now()
    waiting_logged_for_epoch = None

    while not base.shutdown_event.is_set():
        try:
            # Get current blockchain state
            epoch_length = state['epoch_length']
            current_block = await state['subtensor'].get_current_block()
            current_epoch = current_block // epoch_length
            epoch_start_block = current_epoch * epoch_length
            blocks_into_epoch = current_block - epoch_start_block
            blocks_remaining = epoch_length - blocks_into_epoch
            trigger_block = _trigger_block(epoch_length)
            blocks_until_trigger = max(0, trigger_block - blocks_into_epoch)

            # Periodic status logging
            now = datetime.datetime.now()
            if (now - last_status_log).total_seconds() >= base.STATUS_LOG_INTERVAL:
                bt.logging.info(
                    f"📊 Status | Block: {current_block} | Epoch: {current_epoch} | "
                    f"Blocks into epoch: {blocks_into_epoch} | "
                    f"Blocks remaining: {blocks_remaining} | "
                    f"Trigger at block {trigger_block} of the epoch "
                    f"({blocks_until_trigger} to go) | "
                    f"Last acted epoch: {last_acted_epoch}"
                )
                last_status_log = now

            # ==============================================================
            # TRIGGER: this epoch has not been submitted for yet AND it is
            # down to its last LATE_SUBMIT_BLOCKS_REMAINING blocks
            # ==============================================================
            if (
                current_epoch != last_acted_epoch
                and blocks_remaining <= LATE_SUBMIT_BLOCKS_REMAINING
            ):
                if blocks_remaining < LATE_SUBMIT_MIN_BLOCKS:
                    # Too little of the epoch left to finish inside it. A
                    # commit that lands past the boundary is discarded by the
                    # validator and still spends the molecules, so sit this
                    # epoch out and wait for the next one.
                    bt.logging.warning(
                        f"⏭️  Epoch {current_epoch} is already down to "
                        f"{blocks_remaining} block(s) — under the "
                        f"{LATE_SUBMIT_MIN_BLOCKS} needed to finish inside the "
                        f"epoch. A commit landing after the boundary would be "
                        f"discarded AND would spend the molecules, so skipping "
                        f"this epoch and waiting for the next one.\n"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(base.POLL_INTERVAL)
                    continue

                bt.logging.info(
                    f"🟢 Epoch {current_epoch} has {blocks_remaining} block(s) "
                    f"left (block {blocks_into_epoch}/{epoch_length} of the "
                    f"epoch) — submitting now\n"
                )

                await base.do_epoch_submission(state, current_epoch)

                last_acted_epoch = current_epoch
                await asyncio.sleep(base.POLL_INTERVAL)
                continue

            # Not yet in the window — say so once per epoch, then keep polling
            if (
                current_epoch != last_acted_epoch
                and waiting_logged_for_epoch != current_epoch
            ):
                bt.logging.info(
                    f"⏳ Epoch {current_epoch}: waiting for the late-submit "
                    f"window — {blocks_until_trigger} block(s) until block "
                    f"{trigger_block} of the epoch "
                    f"(~{blocks_until_trigger * 12 / 60:.1f} min)"
                )
                waiting_logged_for_epoch = current_epoch

            await asyncio.sleep(base.POLL_INTERVAL)

        except Exception as e:
            bt.logging.error(f"❌ Error in epoch loop: {e}")
            bt.logging.error(traceback.format_exc())
            await asyncio.sleep(10)

    bt.logging.info("\n🛑 Epoch loop terminated by shutdown signal")


# run_miner() looks run_epoch_loop up on the miner module at call time, so
# pointing that name at the late loop is what makes the imported pipeline run
# on this schedule. Everything else in run_miner -- connection setup, the state
# dict, the connection teardown -- is reused untouched.
base.run_epoch_loop = run_epoch_loop


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Entry point: same as miner/submit.py, with the late-trigger banner."""
    load_dotenv()

    config = base.parse_arguments()
    base.setup_logging(config)

    # setup_logging() prints the miner's own banner, which advertises the
    # immediate trigger. Correct the record.
    bt.logging.info("=" * 70)
    bt.logging.info("🕗 LATE-SUBMIT MODE (miner/late_submit.py)")
    bt.logging.info(
        f"⏰ Trigger: {LATE_SUBMIT_BLOCKS_REMAINING} block(s) before the epoch "
        f"ends — block {_trigger_block(base.EPOCH_LENGTH)} of "
        f"{base.EPOCH_LENGTH} — NOT on startup or epoch change"
    )
    bt.logging.info(
        f"🛡️  Skips an epoch with under {LATE_SUBMIT_MIN_BLOCKS} block(s) left "
        f"rather than risk a commit landing in the next epoch"
    )
    bt.logging.info("=" * 70 + "\n")

    try:
        asyncio.run(base.run_miner(config))
    except KeyboardInterrupt:
        bt.logging.info("\n🛑 Miner interrupted by user")
    except Exception as e:
        bt.logging.error(f"❌ Fatal error: {e}")
        bt.logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
