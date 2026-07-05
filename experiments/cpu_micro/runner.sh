#!/bin/bash
cd /home/user/modded-nanogpt/experiments/cpu_micro
B=4000000
python3 train_micro.py --variant gpt2-adamw    --lr 5e-4 --budget $B
python3 train_micro.py --variant gpt2-adamw    --lr 1e-3 --budget $B
python3 train_micro.py --variant gpt2-adamw    --lr 3e-3 --budget $B
python3 train_micro.py --variant modern-adamw  --lr 5e-4 --budget $B
python3 train_micro.py --variant modern-adamw  --lr 1e-3 --budget $B
python3 train_micro.py --variant modern-adamw  --lr 3e-3 --budget $B
python3 train_micro.py --variant modern-muon   --lr 0.02 --budget $B
python3 train_micro.py --variant modern-muon   --lr 0.05 --budget $B
python3 train_micro.py --variant shortcut-muon --lr 0.05 --budget $B
echo ALL_DONE
