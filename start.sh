#!/bin/bash
# Wait for 1 hour and 30 minutes (5400 seconds)
sleep 5400
# Start the PM2 process
pm2 start "python3 neurons/multi_submit.py --wallet.name nova --netuid 68 --network finney --logging.debug" --name multi
