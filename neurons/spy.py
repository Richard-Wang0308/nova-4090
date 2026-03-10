"""
Commitment retrieval and decryption functionality for the validator
CORRECTED VERSION: Collect previous epoch commitments immediately, decrypt immediately
No waiting for epoch-10 block trigger
"""

import asyncio
import hashlib
import requests
from ast import literal_eval
from types import SimpleNamespace
from typing import cast, Optional, Dict, Tuple
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata

# Constants
EPOCH_BLOCKS = 361
MAX_RESPONSE_SIZE = 20 * 1024  # 20KB


@dataclass
class CommitmentData:
  """Stores commitment metadata"""
  uid: int
  hotkey: str
  block: int
  data: str
  timestamp: str = ""


@dataclass
class EpochCommitments:
  """Stores all commitments for an epoch"""
  epoch_number: int
  start_block: int
  end_block: int
  commitments: Dict[int, CommitmentData] = field(default_factory=dict)
  collected_at_block: int = None
  decrypted_at_block: int = None
  decrypted_submissions: Dict[int, list] = None
  push_timestamps: Dict[int, str] = field(default_factory=dict)


async def query_commitment_in_thread(
  subtensor,
  netuid: int,
  hotkey: str,
  block_hash: str
) -> Optional[dict]:
  """
  Query a single commitment in a thread pool to avoid blocking.
  
  Args:
      subtensor: Subtensor instance
      netuid: Network ID
      hotkey: Validator hotkey
      block_hash: Block hash to query at
      
  Returns:
      Commitment dict or None
  """
  def _query():
      try:
          return subtensor.substrate.query(
              module="Commitments",
              storage_function="CommitmentOf",
              params=[netuid, hotkey],
              block_hash=block_hash,
          )
      except Exception as e:
          bt.logging.debug(f"Query failed for {hotkey}: {e}")
          return None
  
  # Run synchronous query in thread pool
  return await asyncio.to_thread(_query)


async def get_commitments(
  subtensor, 
  metagraph, 
  block_hash: str, 
  netuid: int, 
  min_block: int, 
  max_block: int
) -> Dict[str, SimpleNamespace]:
  """
  Retrieve commitments for all validators on a given subnet at a specific block.
  Only returns commitments within the specified block range.

  Args:
      subtensor: The subtensor client object.
      metagraph: The metagraph for the subnet.
      block_hash: The block hash to query at.
      netuid: The network ID.
      min_block: Minimum block for filtering (EXCLUSIVE).
      max_block: Maximum block for filtering (EXCLUSIVE).

  Returns:
      dict: A mapping from hotkey to a SimpleNamespace containing uid, hotkey,
            data (commitment), and block.
  """
  print(f"\n[COLLECTION] Querying commitments at block hash: {block_hash[:20]}...")
  print(f"[COLLECTION] Querying {len(metagraph.hotkeys)} hotkeys...")
  print(f"[COLLECTION] Block range filter: {min_block} < block < {max_block} (exclusive)\n")

  # Gather commitment queries for all validators (hotkeys) concurrently.
  commits = await asyncio.gather(*[
      query_commitment_in_thread(subtensor, netuid, hotkey, block_hash)
      for hotkey in metagraph.hotkeys
  ])

  # Process the results and build a dictionary with additional metadata.
  result = {}
  none_count = 0
  extracted_count = 0
  out_of_range_count = 0
  
  for uid, hotkey in enumerate(metagraph.hotkeys):
      commit = cast(dict, commits[uid])
      
      # Handle None (validator hasn't submitted)
      if commit is None:
          none_count += 1
          continue
      
      # Extract block number
      block_num = commit.get('block')
      if block_num is None:
          continue
      
      # Filter by block range (EXCLUSIVE: > and <)
      # This ensures we only get commitments from the target epoch
      if not (min_block < block_num < max_block):
          out_of_range_count += 1
          continue
      
      # Decode the commitment data using decode_metadata
      try:
          decoded_data = decode_metadata(commit)
          result[hotkey] = SimpleNamespace(
              uid=uid,
              hotkey=hotkey,
              block=block_num,
              data=decoded_data
          )
          extracted_count += 1
          bt.logging.debug(f"  UID {uid}: block={block_num}, data={str(decoded_data)[:60]}...")
      except Exception as e:
          bt.logging.debug(f"  UID {uid}: decode error {e}")
          continue
  
  print(f"[COLLECTION] ════════════════════════════════════════")
  print(f"[COLLECTION] Summary:")
  print(f"[COLLECTION]   Total hotkeys queried: {len(metagraph.hotkeys)}")
  print(f"[COLLECTION]   None (not submitted): {none_count}")
  print(f"[COLLECTION]   Out of range: {out_of_range_count}")
  print(f"[COLLECTION]   Successfully extracted: {extracted_count}")
  print(f"[COLLECTION]   In block range: {len(result)}")
  print(f"[COLLECTION] ════════════════════════════════════════\n")
  
  return result


def tuple_safe_eval(input_str: str) -> Optional[Tuple[int, bytes]]:
  """
  Safely deserialize encrypted tuple with validation.
  
  Args:
      input_str: String representation of tuple
      
  Returns:
      Valid tuple (round_num, encrypted_bytes) or None if invalid
  """
  # Limit input size to prevent overly large inputs.
  if len(input_str) > MAX_RESPONSE_SIZE:
      bt.logging.error("Input exceeds allowed size")
      return None
  
  try:
      # Safely evaluate the input string as a Python literal.
      result = literal_eval(input_str)
  except (SyntaxError, ValueError, MemoryError, RecursionError, TypeError) as e:
      bt.logging.error(f"Input is not a valid literal: {e}")
      return None

  # Check that the result is a tuple with exactly two elements.
  if not (isinstance(result, tuple) and len(result) == 2):
      bt.logging.error("Expected a tuple with exactly two elements")
      return None

  # Verify that the first element is an int.
  if not isinstance(result[0], int):
      bt.logging.error("First element must be an int")
      return None
  
  # Verify that the second element is a bytes object.
  if not isinstance(result[1], bytes):
      bt.logging.error("Second element must be a bytes object")
      return None
  
  return result


def decrypt_submissions(
  current_commitments: Dict[str, SimpleNamespace], 
  github_headers: dict, 
  btd, 
  config: dict
) -> Tuple[Dict[int, list], Dict[int, str]]:
  """
  Fetch GitHub submissions and file-specific commit timestamps, then decrypt.
  
  Args:
      current_commitments: Dict mapping hotkey to SimpleNamespace with commitment data
      github_headers: GitHub API headers for authentication
      btd: BittensorDecryptor instance
      config: Configuration dict with 'num_molecules' key
  
  Returns:
      Tuple of (decrypted_submissions, push_timestamps)
      - decrypted_submissions: Dict[uid] -> list of decrypted molecules
      - push_timestamps: Dict[uid] -> GitHub push timestamp
  """
  print("[METADATA] Fetching GitHub metadata for commitments...")
  
  # Extract file paths from commitments
  file_paths = [
      commit.data for commit in current_commitments.values() 
      if '/' in commit.data
  ]
  
  if not file_paths:
      print("[METADATA] No file paths to fetch")
      return {}, {}
  
  print(f"[METADATA] Found {len(set(file_paths))} unique file paths")
  
  # Fetch GitHub data for each unique path
  github_data = {}
  for path in set(file_paths):
      content_url = f"https://raw.githubusercontent.com/{path}"
      try:
          resp = requests.get(
              content_url, 
              headers={**github_headers, "Range": f"bytes=0-{MAX_RESPONSE_SIZE}"},
              timeout=10
          )
          content = resp.content if resp.status_code in [200, 206] else None
          if content is None:
              bt.logging.warning(f"Failed to fetch content: {resp.status_code} for {content_url}")
      except Exception as e:
          bt.logging.warning(f"Error fetching content for {content_url}: {e}")
          content = None
      
      # Only fetch timestamp if content was successful
      timestamp = ''
      if content is not None:
          parts = path.split('/')
          if len(parts) >= 4:
              api_url = f"https://api.github.com/repos/{parts[0]}/{parts[1]}/commits"
              try:
                  resp = requests.get(
                      api_url, 
                      params={'path': '/'.join(parts[3:]), 'per_page': 1}, 
                      headers=github_headers,
                      timeout=10
                  )
                  commits = resp.json() if resp.status_code == 200 else []
                  timestamp = commits[0]['commit']['committer']['date'] if commits else ''
                  if not timestamp:
                      bt.logging.warning(f"No commit history found for {path}")
              except Exception as e:
                  bt.logging.warning(f"Error fetching timestamp for {path}: {e}")
      
      github_data[path] = {'content': content, 'timestamp': timestamp}
  
  # Prepare encrypted submissions for decryption
  encrypted_submissions = {}
  push_timestamps = {}
  
  for commit in current_commitments.values():
      data = github_data.get(commit.data)
      if not data:
          continue
      
      content = data.get('content')
      push_timestamps[commit.uid] = data.get('timestamp', '')
      
      if not content:
          continue
      
      try:
          # Verify file integrity with hash
          content_hash = hashlib.sha256(
              content.decode('utf-8').encode('utf-8')
          ).hexdigest()[:20]
          
          if commit.data.endswith(f'/{content_hash}.txt'):
              # Parse the encrypted tuple from file content
              encrypted_content = tuple_safe_eval(content.decode('utf-8', errors='replace'))
              if encrypted_content:
                  encrypted_submissions[commit.uid] = encrypted_content
                  bt.logging.debug(f"  UID {commit.uid}: Prepared for decryption")
      except Exception as e:
          bt.logging.debug(f"  UID {commit.uid}: Failed to prepare: {e}")
          pass
  
  print(f"[METADATA] Prepared {len(encrypted_submissions)} submissions for decryption")
  
  # Decrypt all submissions
  decrypted_submissions = {}
  try:
      decrypted_raw = btd.decrypt_dict(encrypted_submissions)
      
      # Parse decrypted content (comma-separated molecules)
      decrypted_submissions = {
          k: v.split(',') 
          for k, v in decrypted_raw.items() 
          if v is not None
      }
      
      # Ensure each UID has the correct number of molecules
      num_molecules = config.get('num_molecules', 1)
      decrypted_submissions = {
          k: v 
          for k, v in decrypted_submissions.items() 
          if len(v) == num_molecules
      }
      
      print(f"[METADATA] Successfully decrypted {len(decrypted_submissions)} submissions")
  except Exception as e:
      bt.logging.error(f"Failed to decrypt submissions: {e}")
      decrypted_submissions = {}
  
  bt.logging.info(
      f"GitHub: {len(file_paths)} paths → {len(encrypted_submissions)} encrypted → "
      f"{len(decrypted_submissions)} decrypted"
  )
  
  return decrypted_submissions, push_timestamps


class EpochCommitmentCollector:
  """
  Manages commitment collection and decryption.
  
  SIMPLIFIED APPROACH:
  - Collect previous epoch commitments immediately (no waiting)
  - Decrypt them immediately
  - Filter by epoch only (ignore current epoch)
  
  Reference: $CITE_1 - Subtensor Storage Query Examples
  """
  
  def __init__(self, subtensor, metagraph, netuid: int, github_headers: dict):
      """
      Args:
          subtensor: Subtensor instance (synchronous)
          metagraph: Metagraph for the subnet
          netuid: Subnet UID
          github_headers: GitHub API headers for authentication
      """
      self.subtensor = subtensor
      self.metagraph = metagraph
      self.netuid = netuid
      self.github_headers = github_headers
      
      # Store collected and decrypted epochs
      self.collected_epochs: Dict[int, EpochCommitments] = {}
      self.last_collected_epoch = -1
      
  def get_current_epoch_info(self, block: int) -> Tuple[int, int, int]:
      """
      Calculate current epoch number and blocks until next epoch.
      
      Args:
          block: Current block number
          
      Returns:
          (epoch_number, blocks_in_current_epoch, blocks_until_next_epoch)
      """
      epoch_number = block // EPOCH_BLOCKS
      blocks_in_current_epoch = block % EPOCH_BLOCKS
      blocks_until_next_epoch = EPOCH_BLOCKS - blocks_in_current_epoch
      
      return epoch_number, blocks_in_current_epoch, blocks_until_next_epoch
  
  async def collect_and_decrypt_previous_epoch(self, current_block: int, btd, config: dict):
      """
      Collect commitments from PREVIOUS epoch and decrypt immediately.
      
      SIMPLIFIED TIMING:
      - Get current epoch
      - Collect from previous epoch (epoch_num - 1)
      - Decrypt immediately
      - No waiting for specific blocks
      
      Args:
          current_block: Current block number
          btd: BittensorDecryptor instance
          config: Configuration dict with 'num_molecules'
      """
      epoch_num, blocks_in_epoch, blocks_until_next = self.get_current_epoch_info(current_block)
      
      # Calculate which epoch to collect from
      target_epoch = epoch_num - 1
      
      # Skip if we already collected this epoch
      if target_epoch in self.collected_epochs:
          return
      
      # Skip if target epoch is negative (we're in epoch 0)
      if target_epoch < 0:
          return
      
      print(f"\n{'='*70}")
      print(f"[BLOCK {current_block}] ✓✓✓ COLLECTING EPOCH {target_epoch}")
      print(f"[BLOCK {current_block}] Current epoch: {epoch_num}")
      print(f"[BLOCK {current_block}] Collecting from previous epoch: {target_epoch}")
      print(f"{'='*70}\n")
      
      try:
          # Get block hash for current block
          block_hash = await asyncio.to_thread(
              self.subtensor.determine_block_hash,
              current_block
          )
          
          if not block_hash:
              print(f"[BLOCK {current_block}] Failed to get block hash")
              return
          
          # Calculate block range for the target epoch
          epoch_start_block = target_epoch * EPOCH_BLOCKS
          epoch_end_block = (target_epoch + 1) * EPOCH_BLOCKS
          
          print(f"[BLOCK {current_block}] Querying Epoch {target_epoch} block range: {epoch_start_block} to {epoch_end_block-1}")
          
          # Collect commitments (EXCLUSIVE range: min_block < block < max_block)
          commitments_dict = await get_commitments(
              self.subtensor,
              self.metagraph,
              block_hash=block_hash,
              netuid=self.netuid,
              min_block=epoch_start_block,
              max_block=epoch_end_block
          )
          
          if not commitments_dict:
              print(f"[BLOCK {current_block}] No commitments found for epoch {target_epoch}")
              self.collected_epochs[target_epoch] = EpochCommitments(
                  epoch_number=target_epoch,
                  start_block=epoch_start_block,
                  end_block=epoch_end_block,
                  commitments={},
                  collected_at_block=current_block
              )
              return
          
          # Convert to CommitmentData objects
          commitments = {}
          for uid, (hotkey, commit_ns) in enumerate(commitments_dict.items()):
              commitments[commit_ns.uid] = CommitmentData(
                  uid=commit_ns.uid,
                  hotkey=commit_ns.hotkey,
                  block=commit_ns.block,
                  data=commit_ns.data
              )
          
          # Store collected commitments
          epoch_commitments = EpochCommitments(
              epoch_number=target_epoch,
              start_block=epoch_start_block,
              end_block=epoch_end_block,
              commitments=commitments,
              collected_at_block=current_block
          )
          
          self.collected_epochs[target_epoch] = epoch_commitments
          
          print(f"[BLOCK {current_block}] ✓ Collected {len(commitments)} commitments from epoch {target_epoch}")
          print(f"[BLOCK {current_block}] Starting decryption...\n")
          
          # Decrypt immediately
          await self.decrypt_epoch(target_epoch, btd, config)
          
      except Exception as e:
          print(f"[BLOCK {current_block}] Collection failed: {e}")
          import traceback
          traceback.print_exc()
  
  async def decrypt_epoch(self, epoch_num: int, btd, config: dict):
      """
      Decrypt commitments for a specific epoch.
      
      Args:
          epoch_num: Epoch number to decrypt
          btd: BittensorDecryptor instance
          config: Configuration dict with 'num_molecules'
      """
      if epoch_num not in self.collected_epochs:
          return
      
      epoch_commitments = self.collected_epochs[epoch_num]
      
      if not epoch_commitments.commitments:
          print(f"[DECRYPT] No commitments to decrypt for epoch {epoch_num}")
          return
      
      if epoch_commitments.decrypted_at_block is not None:
          return  # Already decrypted
      
      print(f"\n{'='*70}")
      print(f"[DECRYPT] ✓✓✓ DECRYPTING EPOCH {epoch_num}")
      print(f"{'='*70}\n")
      
      # Convert CommitmentData back to SimpleNamespace for decrypt_submissions
      current_commitments = {}
      for uid, commit in epoch_commitments.commitments.items():
          current_commitments[commit.hotkey] = SimpleNamespace(
              uid=commit.uid,
              hotkey=commit.hotkey,
              block=commit.block,
              data=commit.data
          )
      
      # Decrypt (run in thread pool to avoid blocking)
      try:
          decrypted_submissions, push_timestamps = await asyncio.to_thread(
              decrypt_submissions,
              current_commitments,
              self.github_headers,
              btd,
              config
          )
          
          epoch_commitments.decrypted_submissions = decrypted_submissions
          epoch_commitments.push_timestamps = push_timestamps
          epoch_commitments.decrypted_at_block = (
              self.subtensor.get_current_block()
          )
          
          print(f"[DECRYPT] ✓ Decryption complete for epoch {epoch_num}\n")
          
          # Display results
          await self.display_decrypted_content(epoch_num)
          
      except Exception as e:
          print(f"[DECRYPT] Decryption failed for epoch {epoch_num}: {e}")
          import traceback
          traceback.print_exc()
  
  async def display_decrypted_content(self, epoch_num: int):
      """
      Display all decrypted submissions with metadata.
      
      Args:
          epoch_num: Epoch number to display
      """
      if epoch_num not in self.collected_epochs:
          return
      
      epoch_commitments = self.collected_epochs[epoch_num]
      
      if not epoch_commitments.decrypted_submissions:
          bt.logging.warning(f"[DISPLAY] No decrypted submissions for epoch {epoch_num}")
          return
      
      print(f"{'='*70}")
      print(f"[DISPLAY] DECRYPTED SUBMISSIONS FROM EPOCH {epoch_num}")
      print(f"{'='*70}\n")
      
      decrypted_count = 0
      for uid, molecules in epoch_commitments.decrypted_submissions.items():
          commit = epoch_commitments.commitments.get(uid)
          push_time = epoch_commitments.push_timestamps.get(uid, '')
          
          if commit and molecules is not None:
              decrypted_count += 1
              print(f"UID {uid}:")
              print(f"  Hotkey: {commit.hotkey[:20]}...")
              print(f"  Block Submitted: {commit.block}")
              print(f"  Push Timestamp: {push_time}")
              print(f"  Decrypted Molecules: {molecules}")
              print("")
      
      print(f"{'='*70}")
      print(f"Total decrypted: {decrypted_count}/{len(epoch_commitments.commitments)}")
      print(f"{'='*70}\n")
  
  async def monitor_and_process(self, btd, config: dict, duration_blocks: int = 400):
      """
      Main monitoring loop - collect and decrypt previous epoch immediately.
      
      Args:
          btd: BittensorDecryptor instance
          config: Configuration dict with 'num_molecules'
          duration_blocks: How many blocks to monitor
      """
      start_block = self.subtensor.get_current_block()
      end_block = start_block + duration_blocks
      
      print(f"\n[MONITOR] Starting block monitor from {start_block} to {end_block}")
      print(f"[MONITOR] Epoch duration: {EPOCH_BLOCKS} blocks")
      print(f"[MONITOR] Block time: 12 seconds per block")
      print(f"[MONITOR]")
      print(f"[MONITOR] SIMPLIFIED APPROACH:")
      print(f"[MONITOR]   - Collect previous epoch commitments immediately")
      print(f"[MONITOR]   - Decrypt immediately (no waiting)")
      print(f"[MONITOR]   - Filter by epoch only (ignore current epoch)")
      print(f"[MONITOR]\n")
      
      last_block = start_block - 1
      
      while True:
          current_block = self.subtensor.get_current_block()
          
          if current_block >= end_block:
              break
          
          # Skip if block hasn't changed
          if current_block == last_block:
              await asyncio.sleep(1)
              continue
          
          # Block advanced - process all blocks between last_block and current_block
          for block_to_process in range(last_block + 1, current_block + 1):
              epoch_num, blocks_in_epoch, blocks_until_next = self.get_current_epoch_info(block_to_process)
              
              # Display status every 10 blocks
              if block_to_process % 10 == 0:
                  collected_epochs = len(self.collected_epochs)
                  print(
                      f"[BLOCK {block_to_process}] Epoch {epoch_num}, "
                      f"Block {blocks_in_epoch}/{EPOCH_BLOCKS}, "
                      f"Collected epochs: {collected_epochs}"
                  )
              
              # Collect and decrypt previous epoch immediately
              await self.collect_and_decrypt_previous_epoch(block_to_process, btd, config)
          
          last_block = current_block
          await asyncio.sleep(1)
      
      print(f"\n[MONITOR] Monitoring complete at block {current_block}\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def collect_commitments(
  netuid: int = 68,
  network: Optional[str] = None,
  duration_blocks: int = 400,
) -> Optional[Dict[int, EpochCommitments]]:
  """
  Collect commitments with simplified immediate approach.
  
  Args:
      netuid: The network ID
      network: Bittensor network name
      duration_blocks: How many blocks to monitor
      
  Returns:
      Dict of epoch_number -> EpochCommitments
  """
  load_dotenv()
  
  # Get network from env or arg
  network = network or os.environ.get("SUBTENSOR_NETWORK", "finney")
  
  # GitHub authentication
  github_token = os.environ.get("GITHUB_TOKEN", "")
  github_headers = {
      "Authorization": f"token {github_token}" if github_token else "",
      "Accept": "application/vnd.github.v3+json"
  }
  
  # Initialize Subtensor (SYNCHRONOUS)
  subtensor = bt.subtensor(network=network)
  
  try:
      print(f"[INIT] Connected to {network} network\n")
      
      # Get metagraph (SYNCHRONOUS)
      metagraph = subtensor.metagraph(netuid)
      print(f"[INIT] Loaded metagraph for subnet {netuid} with {len(metagraph.hotkeys)} validators\n")
      
      # Mock BittensorDecryptor (replace with actual implementation)
      class MockBittensorDecryptor:
          def decrypt_dict(self, encrypted_dict):
              """Mock decryption - replace with real decryption logic"""
              decrypted = {}
              for uid, (round_num, encrypted_bytes) in encrypted_dict.items():
                  try:
                      # This is a mock - real implementation uses Drand
                      decrypted[uid] = f"DECRYPTED_CONTENT_UID_{uid}"
                  except Exception as e:
                      print(f"Failed to decrypt UID {uid}: {e}")
                      decrypted[uid] = None
              return decrypted
      
      btd = MockBittensorDecryptor()
      
      # Configuration
      config = {
          'num_molecules': 1
      }
      
      # Create collector
      collector = EpochCommitmentCollector(
          subtensor=subtensor,
          metagraph=metagraph,
          netuid=netuid,
          github_headers=github_headers
      )
      
      # Monitor for specified blocks
      await collector.monitor_and_process(btd, config, duration_blocks=duration_blocks)
      
      return collector.collected_epochs
      
  except Exception as e:
      print(f"[ERROR] Error during collection: {e}")
      import traceback
      traceback.print_exc()
      return None


async def main():
  """Main entry point"""
  import argparse
  
  parser = argparse.ArgumentParser(
      description='Collect commitments from Bittensor subnet'
  )
  parser.add_argument(
      '--netuid',
      type=int,
      default=68,
      help='Subnet netuid (default: 68)'
  )
  parser.add_argument(
      '--network',
      type=str,
      default=None,
      help='Bittensor network (defaults to SUBTENSOR_NETWORK env var or finney)'
  )
  parser.add_argument(
      '--duration_blocks',
      type=int,
      default=400,
      help='Number of blocks to monitor (default: 400)'
  )
  parser.add_argument(
      '--debug',
      action='store_true',
      help='Enable debug logging'
  )
  
  args = parser.parse_args()
  
  # Setup logging
  config = bt.config()
  if args.debug:
      config.logging.debug = True
  bt.logging(config=config)
  
  # Run collection
  result = await collect_commitments(
      netuid=args.netuid,
      network=args.network,
      duration_blocks=args.duration_blocks
  )
  
  if result:
      total_commitments = sum(len(ec.commitments) for ec in result.values())
      print(f"✓ Collection complete: {len(result)} epochs, {total_commitments} total commitments")
      for epoch_num, epoch_commitments in result.items():
          print(f"  Epoch {epoch_num}: {len(epoch_commitments.commitments)} commitments")
  else:
      print("✗ Collection failed")


if __name__ == "__main__":
  asyncio.run(main())