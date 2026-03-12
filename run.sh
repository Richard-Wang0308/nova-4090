pm2 start "python3 neurons/miner_ban_mini.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug" --name "miner_ban_mini"
pm2 start "python3 neurons/miner_ban_synthon.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug" --name "miner_ban_synthon"
pm2 start "python3 neurons/miner_ban_random_mutate_db.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug" --name "miner_ban_random_mutate"


#crossover

#crossover with db
python3 neurons/miner_ban_mini_db.py --wallet.name xova --wallet.hotkey xotb --logging.debug

#synthon
python3 neurons/miner_ban_synthon_db.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug

#neighbour mutate
python3 neurons/miner_ban_neighbour_mutate.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug

#random mutate
python3 neurons/miner_ban_random_mutate_db.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug

#simple submit
python3 neurons/simple_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotg --logging.debug
python3 neurons/simple_submit.py --wallet.name nova --wallet.hotkey nota --logging.debug
python3 neurons/simple_submit.py --wallet.name xova --wallet.hotkey xota --logging.debug

#top submit
python3 neurons/top_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug

python3 neurons/synthon_miner.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug

python3 neurons/mini_data.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug



CUDA_VISIBLE_DEVICES=0 pm2 start "python3 neurons/advanced_data.py --logging.debug" --name "advanced_data"
CUDA_VISIBLE_DEVICES=0 pm2 start "python3 neurons/mini_data.py --logging.debug" --name "Mini_0"
CUDA_VISIBLE_DEVICES=0 pm2 start "python3 neurons/synthon_data.py --logging.debug" --name "synthon"
CUDA_VISIBLE_DEVICES=1 pm2 start "python3 neurons/mini_data.py --logging.debug" --name "Mini_1"
CUDA_VISIBLE_DEVICES=1 pm2 start "python3 neurons/random_data.py --logging.debug" --name "Random"


CUDA_VISIBLE_DEVICES=1 pm2 start "python3 neurons/synthon_data.py --logging.debug" --name "synthon_data"



