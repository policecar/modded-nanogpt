#!/bin/bash
cd /home/user/modded-nanogpt/experiments/cpu_micro
until grep -q ALL_DONE runner.log 2>/dev/null; do sleep 5; done
python3 train_micro.py --variant gpt2-adamw   --lr 5e-4 --budget 4000000
python3 train_micro.py --variant modern-adamw --lr 5e-4 --budget 4000000
echo EXTRA_DONE
