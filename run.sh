pm2 start "python3 neurons/miner_mini1.py --wallet.name multisig-jjpes-shib --wallet.hotkey hote --logging.debug" --name "miner-mini1"
pm2 start "python3 neurons/miner_mini2.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotf --logging.debug" --name "miner-mini2"


python3 neurons/simple_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug

pm2 start "python3 neurons/miner_mini2.py --wallet.name multisig-jjpes-shib --wallet.hotkey hote --gpu_ids '0,1' --logging.debug" --name "rxn1_miner"

CUDA_VISIBLE_DEVICES=0,1 pm2 start "python3 neurons/miner_mini2.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotf --gpu_ids '1' --logging.debug" --name "rxn2_miner"

CUDA_VISIBLE_DEVICES=0 pm2 start "python3 neurons/miner_mini1.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotc --logging.debug" --name "rxn5_miner"

CUDA_VISIBLE_DEVICES=1 pm2 start "python3 neurons/miner_mini1.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotb --logging.debug" --name "rxn5_miner"

CUDA_VISIBLE_DEVICES=0 python3 neurons/miner_mini1.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotc --logging.debug

CUDA_VISIBLE_DEVICES=0 python3 neurons/simple_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotc --logging.debug


CUDA_VISIBLE_DEVICES=0 python3 neurons/simple_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hote --logging.debug
CUDA_VISIBLE_DEVICES=0 python3 neurons/simple_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug
CUDA_VISIBLE_DEVICES=0 python3 neurons/simple_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotb --logging.debug
CUDA_VISIBLE_DEVICES=0 python3 neurons/simple_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hota --logging.debug

