pm2 start "python3 neurons/miner_ban_mini.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotb --logging.debug" --name "miner_ban_mini"
pm2 start "python3 neurons/miner_ban_synthon.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotb --logging.debug" --name "miner_ban_synthon"
pm2 start "python3 neurons/miner_ban_random_mutate.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotb --logging.debug" --name "miner_ban_random_mutate"


#crossover
python3 neurons/miner_ban_mini.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotb --logging.debug

#crossover with db
python3 neurons/miner_ban_mini_db.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotb --logging.debug

#synthon
python3 neurons/miner_ban_synthon.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotb --logging.debug

#neighbour mutate
python3 neurons/miner_ban_neighbour_mutate.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotb --logging.debug

#random mutate
python3 neurons/miner_ban_random_mutate.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotb --logging.debug

#simple submit
python3 neurons/simple_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hota --logging.debug

#top submit
python3 neurons/top_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hota --logging.debug
