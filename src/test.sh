#!/bin/bash
runs="python3 server_ird.py is_wandb=False env=serving_imit"

a="[1.0,1.0,0,0.1]"
b="[1.0,1.0,0,0.3]"
c="[1.0,1.0,0,0.5]"
tmux new-session -d -s "1"
tmux new-session -d -s "3"
tmux new-session -d -s "5"
# tmux new-session -d -s "$d"
# tmux new-session -d -s "$e"
# tmux new-session -d -s "$f"
sleep 1

tmux send-keys -t "1" "conda activate test; cd ~/source/imitation/src" enter
tmux send-keys -t "3" "conda activate test; cd ~/source/imitation/src" enter
tmux send-keys -t "5" "conda activate test; cd ~/source/imitation/src" enter
# # tmux send-keys -t "$d" "conda activate CI" enter
# tmux send-keys -t "$e" "conda activate CI" enter
# tmux send-keys -t "$f" "conda activate CI" enter

sleep 1
tmux send-keys -t "1" "$runs disc.coef=$a " enter
sleep 5
tmux send-keys -t "3" "$runs disc.coef=$b " enter
sleep 5
tmux send-keys -t "5" "$runs disc.coef=$c " enter
# sleep 5
# tmux send-keys -t "$d" "$runs2 agent.actor_lr=3e-5" enter
# sleep 5
# tmux send-keys -t "$e" "$runs3 agent.actor_lr=3e-4" enter
# sleep 5
# tmux send-keys -t "$f" "$runs3 agent.actor_lr=3e-5" enter
