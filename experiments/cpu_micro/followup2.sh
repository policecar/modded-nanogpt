#!/bin/bash
cd /home/user/modded-nanogpt/experiments/cpu_micro
until grep -q ORIGINAL_DONE runner.log 2>/dev/null; do sleep 10; done
python3 train_micro.py --variant shortcut-muon --lr 0.02 --budget 4000000
echo EXTRA2_DONE
