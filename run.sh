pm2 start "python3 neurons/miner_ban_mini.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotf --logging.debug" --name "ban_atom"

python3 neurons/miner_ban_mini.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotf --logging.debug

python3 neurons/simple_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hota --logging.debug
