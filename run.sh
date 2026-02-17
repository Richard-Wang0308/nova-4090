pm2 start "python3 neurons/miner_ban_mini.py --wallet.name multisig-jjpes-shib --wallet.hotkey hota --logging.debug" --name "miner_ban_mini"
python3 neurons/miner_ban_mini.py --wallet.name multisig-jjpes-shib --wallet.hotkey hota --logging.debug
python3 neurons/miner_ban_synthon.py --wallet.name multisig-jjpes-shib --wallet.hotkey hota --logging.debug
python3 neurons/miner_ban_neighbour_mutate.py --wallet.name multisig-jjpes-shib --wallet.hotkey hota --logging.debug
python3 neurons/miner_ban_random_mutate.py --wallet.name multisig-jjpes-shib --wallet.hotkey hota --logging.debug
python3 neurons/simple_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hota --logging.debug
