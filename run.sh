#crossover with db
python3 neurons/miner_ban_mini_db.py --wallet.name xova --wallet.hotkey xotb --logging.debug

#synthon
python3 neurons/miner_ban_synthon_db.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug

#neighbour mutate
python3 neurons/miner_ban_neighbour_mutate.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug

#random mutate
python3 neurons/miner_ban_random_mutate_db.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug

#simple submit
python3 neurons/simple_submit.py --wallet.name nova --wallet.hotkey notc --logging.debug
python3 neurons/simple_submit.py --wallet.name nova --wallet.hotkey notd --logging.debug
python3 neurons/simple_submit.py --wallet.name nova --wallet.hotkey note --logging.debug


#top submit
python3 neurons/top_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug

python3 neurons/synthon_miner.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug

python3 neurons/mini_data.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug


CUDA_VISIBLE_DEVICES=1 pm2 start "python3 neurons/synthon_data.py --logging.debug" --name "synthon_data"

pm2 start "python3 neurons/repeat_submit_one.py --wallet.name nova --wallet.hotkey notc --netuid 68 --network finney" --name one
pm2 start "python3 neurons/repeat_submit_two.py --wallet.name nova --wallet.hotkey notd --netuid 68 --network finney" --name two
pm2 start "python3 neurons/repeat_submit_three.py --wallet.name nova --wallet.hotkey note --netuid 68 --network finney" --name three
