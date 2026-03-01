import asyncio
import hashlib
import requests
import bittensor as bt
from ast import literal_eval
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import os
from dotenv import load_dotenv

# Constants
EPOCH_BLOCKS = 361  # Actual Bittensor epoch duration
COLLECTION_BLOCKS_BEFORE_EPOCH = 10
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

class EpochCommitmentCollector:
    """
    Manages commitment collection at epoch-10 blocks and decryption at epoch change.
    
    UPDATED: Proper AsyncSubtensor lifecycle management with initialization and cleanup.
    """
    
    def __init__(self, async_subtensor, metagraph, netuid: int, github_headers: dict):
        """
        Args:
            async_subtensor: AsyncSubtensor instance (already initialized)
            metagraph: Metagraph for the subnet
            netuid: Subnet UID
            github_headers: GitHub API headers for authentication
        """
        self.async_subtensor = async_subtensor
        self.metagraph = metagraph
        self.netuid = netuid
        self.github_headers = github_headers
        self.current_epoch_commitments: Optional[EpochCommitments] = None
        self.collected = False
        self.last_block = 0
        
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
    
    async def get_commitments(
        self, 
        block_hash: str,
        min_block: int,
        max_block: int
    ) -> Dict[int, CommitmentData]:
        """
        Retrieve commitments from blockchain for all validators concurrently.
        
        Args:
            block_hash: Hash of the block to query
            min_block: Minimum block for filtering
            max_block: Maximum block for filtering
            
        Returns:
            Dictionary mapping UID to CommitmentData
        """
        bt.logging.info(f"[COLLECTION] Querying commitments for {len(self.metagraph.hotkeys)} hotkeys...")
        
        try:
            # Query commitments for all hotkeys concurrently
            commits = await asyncio.gather(*[
                self.async_subtensor.substrate.query(
                    module="Commitments",
                    storage_function="CommitmentOf",
                    params=[self.netuid, hotkey],
                    block_hash=block_hash,
                ) for hotkey in self.metagraph.hotkeys
            ])
            
            commitments_dict = {}
            for uid, (hotkey, commit_data) in enumerate(zip(self.metagraph.hotkeys, commits)):
                if commit_data and commit_data.value is not None:
                    try:
                        # Decode the commitment data
                        decoded = self.async_subtensor.decode_params(
                            "Commitments", 
                            "CommitmentOf", 
                            commit_data.value
                        )
                        
                        if decoded and len(decoded) >= 2:
                            block_num = decoded[0]
                            commit_info = decoded[1]
                            
                            # Filter by block range
                            if min_block < block_num < max_block:
                                commitments_dict[uid] = CommitmentData(
                                    uid=uid,
                                    hotkey=hotkey,
                                    block=block_num,
                                    data=str(commit_info)
                                )
                                bt.logging.debug(f"  UID {uid}: block={block_num}")
                    except Exception as e:
                        bt.logging.debug(f"  UID {uid}: decode error {e}")
            
            bt.logging.info(f"[COLLECTION] Found {len(commitments_dict)} commitments")
            return commitments_dict
            
        except Exception as e:
            bt.logging.error(f"[COLLECTION] Query failed: {e}")
            return {}
    
    def tuple_safe_eval(self, input_str: str) -> Optional[Tuple[int, bytes]]:
        """
        Safely deserialize encrypted tuple with validation.
        
        Args:
            input_str: String representation of tuple
            
        Returns:
            Valid tuple or None if invalid
        """
        if len(input_str) > MAX_RESPONSE_SIZE:
            bt.logging.error(f"Input exceeds {MAX_RESPONSE_SIZE} bytes")
            return None
        
        try:
            result = literal_eval(input_str)
        except (SyntaxError, ValueError, MemoryError, RecursionError, TypeError) as e:
            bt.logging.error(f"Failed to parse tuple: {e}")
            return None
        
        # Strict validation
        if not (isinstance(result, tuple) and len(result) == 2):
            bt.logging.error(f"Invalid tuple structure: {result}")
            return None
        if not isinstance(result[0], int):
            bt.logging.error(f"First element not int: {type(result[0])}")
            return None
        if not isinstance(result[1], bytes):
            bt.logging.error(f"Second element not bytes: {type(result[1])}")
            return None
        
        return result
    
    async def fetch_github_metadata(
        self, 
        commitments: Dict[int, CommitmentData]
    ) -> Dict[int, Dict]:
        """
        Fetch GitHub metadata (timestamps, content hashes) for all commitments concurrently.
        
        Args:
            commitments: Dictionary of commitments
            
        Returns:
            Dictionary mapping UID to GitHub metadata
        """
        bt.logging.info("[METADATA] Fetching GitHub metadata for commitments...")
        
        github_data = {}
        file_paths = set()
        
        # Extract unique file paths
        for commit in commitments.values():
            if '/' in commit.data:
                file_paths.add(commit.data)
        
        if not file_paths:
            bt.logging.info("[METADATA] No file paths to fetch")
            return {}
        
        # Fetch content and timestamps concurrently
        async def fetch_single_path(path: str):
            try:
                # Fetch raw content
                content_url = f"https://raw.githubusercontent.com/{path}"
                resp = requests.get(
                    content_url,
                    headers={**self.github_headers, "Range": f"bytes=0-{MAX_RESPONSE_SIZE}"},
                    timeout=10
                )
                
                content = resp.content if resp.status_code in [200, 206] else None
                
                # Fetch commit timestamp
                timestamp = ''
                if content is not None:
                    parts = path.split('/')
                    if len(parts) >= 2:
                        api_url = f"https://api.github.com/repos/{parts[0]}/{parts[1]}/commits"
                        resp_api = requests.get(
                            api_url,
                            params={'path': '/'.join(parts[3:]), 'per_page': 1},
                            headers=self.github_headers,
                            timeout=10
                        )
                        commits_list = resp_api.json() if resp_api.status_code == 200 else []
                        timestamp = commits_list[0]['commit']['committer']['date'] if commits_list else ''
                
                return path, {'content': content, 'timestamp': timestamp}
                
            except Exception as e:
                bt.logging.error(f"Failed to fetch {path}: {e}")
                return path, {'content': None, 'timestamp': ''}
        
        # Fetch all paths concurrently
        fetch_tasks = [fetch_single_path(path) for path in file_paths]
        results = await asyncio.gather(*fetch_tasks)
        
        for path, data in results:
            github_data[path] = data
            bt.logging.debug(f"  Fetched {path}: {len(data['content']) if data['content'] else 0} bytes")
        
        # Map UIDs to GitHub data
        uid_to_github = {}
        for uid, commit in commitments.items():
            if commit.data in github_data:
                uid_to_github[uid] = github_data[commit.data]
        
        bt.logging.info(f"[METADATA] Fetched metadata for {len(uid_to_github)} UIDs")
        return uid_to_github
    
    async def collect_commitments_at_epoch_minus_10(self, current_block: int):
        """
        Collect commitments when 10 blocks remain in current epoch.
        
        Args:
            current_block: Current block number
        """
        epoch_num, blocks_in_epoch, blocks_until_next = self.get_current_epoch_info(current_block)
        
        # Check if we're at the trigger point (10 blocks before epoch end)
        if blocks_until_next != COLLECTION_BLOCKS_BEFORE_EPOCH:
            return
        
        if self.collected:
            bt.logging.debug(f"Already collected for epoch {epoch_num}")
            return
        
        bt.logging.info(f"\n{'='*70}")
        bt.logging.info(f"[EPOCH {epoch_num}] ✓✓✓ COLLECTION TRIGGERED at block {current_block}")
        bt.logging.info(f"[EPOCH {epoch_num}] Blocks until next epoch: {blocks_until_next}")
        bt.logging.info(f"{'='*70}\n")
        
        try:
            # Get block hash using AsyncSubtensor
            block_hash = await self.async_subtensor.substrate.get_block_hash(current_block)
            
            # Calculate block range (full epoch)
            epoch_start_block = epoch_num * EPOCH_BLOCKS
            epoch_end_block = (epoch_num + 1) * EPOCH_BLOCKS
            
            # Collect commitments
            commitments = await self.get_commitments(
                block_hash=block_hash,
                min_block=epoch_start_block,
                max_block=current_block
            )
            
            if not commitments:
                bt.logging.warning(f"[EPOCH {epoch_num}] No commitments found")
                self.collected = True
                return
            
            # Fetch GitHub metadata
            github_data = await self.fetch_github_metadata(commitments)
            
            # Store epoch commitments
            self.current_epoch_commitments = EpochCommitments(
                epoch_number=epoch_num,
                start_block=epoch_start_block,
                end_block=epoch_end_block,
                commitments=commitments,
                collected_at_block=current_block
            )
            
            # Store GitHub metadata in commitments
            for uid, commit in commitments.items():
                if uid in github_data:
                    commit.timestamp = github_data[uid].get('timestamp', '')
            
            self.collected = True
            
            bt.logging.info(f"[EPOCH {epoch_num}] ✓ Collected {len(commitments)} commitments")
            bt.logging.info(f"[EPOCH {epoch_num}] Waiting for epoch change (in {blocks_until_next} blocks)...\n")
            
        except Exception as e:
            bt.logging.error(f"[EPOCH {epoch_num}] Collection failed: {e}")
            import traceback
            traceback.print_exc()
            self.collected = False
    
    async def decrypt_submissions_at_epoch_change(
        self, 
        current_block: int,
        btd  # BittensorDecryptor instance
    ):
        """
        Decrypt submissions immediately when epoch changes.
        
        Args:
            current_block: Current block number
            btd: BittensorDecryptor instance
        """
        if not self.collected:
            return
        
        epoch_num, blocks_in_epoch, blocks_until_next = self.get_current_epoch_info(current_block)
        
        # Check if epoch just changed (we're at block 0 of new epoch)
        if blocks_in_epoch != 0:
            return
        
        if self.current_epoch_commitments.decrypted_at_block is not None:
            return  # Already decrypted
        
        bt.logging.info(f"\n{'='*70}")
        bt.logging.info(f"[EPOCH {epoch_num}] ✓✓✓ EPOCH CHANGE DETECTED at block {current_block}")
        bt.logging.info(f"[EPOCH {epoch_num}] Starting decryption...")
        bt.logging.info(f"{'='*70}\n")
        
        # Prepare encrypted submissions
        encrypted_submissions = {}
        for uid, commit in self.current_epoch_commitments.commitments.items():
            tuple_data = self.tuple_safe_eval(commit.data)
            if tuple_data:
                encrypted_submissions[uid] = tuple_data
        
        bt.logging.info(f"[EPOCH {epoch_num}] Prepared {len(encrypted_submissions)} submissions for decryption")
        
        # Decrypt
        try:
            decrypted = btd.decrypt_dict(encrypted_submissions)
            self.current_epoch_commitments.decrypted_submissions = decrypted
            self.current_epoch_commitments.decrypted_at_block = current_block
            
            bt.logging.info(f"[EPOCH {epoch_num}] ✓ Decryption complete\n")
            
            # Display results
            await self.display_decrypted_content(epoch_num)
            
        except Exception as e:
            bt.logging.error(f"[EPOCH {epoch_num}] Decryption failed: {e}")
    
    async def display_decrypted_content(self, epoch_num: int):
        """
        Display all decrypted submissions with metadata.
        
        Args:
            epoch_num: Epoch number
        """
        if not self.current_epoch_commitments.decrypted_submissions:
            bt.logging.warning(f"[EPOCH {epoch_num}] No decrypted submissions to display")
            return
        
        bt.logging.info(f"{'='*70}")
        bt.logging.info(f"[EPOCH {epoch_num}] DECRYPTED SUBMISSIONS")
        bt.logging.info(f"{'='*70}\n")
        
        decrypted_count = 0
        for uid, decrypted_content in self.current_epoch_commitments.decrypted_submissions.items():
            commit = self.current_epoch_commitments.commitments.get(uid)
            
            if commit and decrypted_content is not None:
                decrypted_count += 1
                bt.logging.info(f"UID {uid}:")
                bt.logging.info(f"  Hotkey: {commit.hotkey[:20]}...")
                bt.logging.info(f"  Block Submitted: {commit.block}")
                bt.logging.info(f"  Push Timestamp: {commit.timestamp}")
                bt.logging.info(f"  Decrypted Content: {decrypted_content}")
                bt.logging.info("")
        
        bt.logging.info(f"{'='*70}")
        bt.logging.info(f"Total decrypted: {decrypted_count}/{len(self.current_epoch_commitments.commitments)}")
        bt.logging.info(f"{'='*70}\n")
    
    async def monitor_and_process(self, btd, duration_blocks: int = 400):
        """
        Main monitoring loop: collect at epoch-10, decrypt at epoch change.
        
        Args:
            btd: BittensorDecryptor instance
            duration_blocks: How many blocks to monitor
        """
        # Get current block using async method
        start_block = await self.async_subtensor.get_current_block()
        end_block = start_block + duration_blocks
        
        bt.logging.info(f"\n[MONITOR] Starting block monitor from {start_block} to {end_block}")
        bt.logging.info(f"[MONITOR] Epoch duration: {EPOCH_BLOCKS} blocks")
        bt.logging.info(f"[MONITOR] Collection trigger: {COLLECTION_BLOCKS_BEFORE_EPOCH} blocks before epoch end")
        bt.logging.info(f"[MONITOR] Block time: 12 seconds per block\n")
        
        last_block = start_block - 1
        
        while True:
            # Get current block using async method
            current_block = await self.async_subtensor.get_current_block()
            
            if current_block >= end_block:
                break
            
            # Skip if block hasn't changed
            if current_block == last_block:
                await asyncio.sleep(1)  # Check every second
                continue
            
            # Block advanced - process all blocks between last_block and current_block
            for block_to_process in range(last_block + 1, current_block + 1):
                epoch_num, blocks_in_epoch, blocks_until_next = self.get_current_epoch_info(block_to_process)
                
                # Display status every 10 blocks
                if block_to_process % 10 == 0:
                    status = "✓ COLLECTED" if self.collected else "⏳ Waiting"
                    bt.logging.info(
                        f"[BLOCK {block_to_process}] Epoch {epoch_num}, "
                        f"Block {blocks_in_epoch}/{EPOCH_BLOCKS}, "
                        f"{blocks_until_next} blocks to next epoch - {status}"
                    )
                
                # Trigger collection at epoch-10
                await self.collect_commitments_at_epoch_minus_10(block_to_process)
                
                # Trigger decryption at epoch change
                await self.decrypt_submissions_at_epoch_change(block_to_process, btd)
            
            last_block = current_block
            await asyncio.sleep(1)  # Check every second
        
        bt.logging.info(f"\n[MONITOR] Monitoring complete at block {current_block}\n")


# ============================================================================
# USAGE EXAMPLE WITH PROPER LIFECYCLE MANAGEMENT
# ============================================================================

async def collect_commitments(
    netuid: int = 68,
    network: Optional[str] = None,
    duration_blocks: int = 400,
) -> Optional[EpochCommitments]:
    """
    Collect commitments with proper AsyncSubtensor lifecycle management.
    
    Args:
        netuid: The network ID
        network: Bittensor network name
        duration_blocks: How many blocks to monitor
        
    Returns:
        EpochCommitments object or None if failed
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
    
    # Initialize AsyncSubtensor with proper lifecycle
    async_subtensor = bt.async_subtensor(network=network)
    
    try:
        # Initialize the async subtensor connection
        await async_subtensor.initialize()
        bt.logging.info(f"Connected to {network} network")
        
        # Get metagraph
        metagraph = await async_subtensor.metagraph(netuid)
        bt.logging.info(f"Loaded metagraph for subnet {netuid} with {len(metagraph.hotkeys)} validators")
        
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
                        bt.logging.error(f"Failed to decrypt UID {uid}: {e}")
                        decrypted[uid] = None
                return decrypted
        
        btd = MockBittensorDecryptor()
        
        # Create collector
        collector = EpochCommitmentCollector(
            async_subtensor=async_subtensor,
            metagraph=metagraph,
            netuid=netuid,
            github_headers=github_headers
        )
        
        # Monitor for specified blocks
        await collector.monitor_and_process(btd, duration_blocks=duration_blocks)
        
        return collector.current_epoch_commitments
        
    except Exception as e:
        bt.logging.error(f"Error during collection: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # CRITICAL: Always close the connection
        try:
            await async_subtensor.close()
            bt.logging.info("AsyncSubtensor connection closed")
        except Exception as e:
            bt.logging.error(f"Error closing connection: {e}")


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
        '--logging.debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    config = bt.config()
    config.logging.debug = args.logging_debug if hasattr(args, 'logging_debug') else False
    bt.logging(config=config)
    
    # Run collection
    result = await collect_commitments(
        netuid=args.netuid,
        network=args.network,
        duration_blocks=args.duration_blocks
    )
    
    if result:
        bt.logging.info(f"✓ Collection complete: {len(result.commitments)} commitments collected")
    else:
        bt.logging.error("✗ Collection failed")


if __name__ == "__main__":
    asyncio.run(main())