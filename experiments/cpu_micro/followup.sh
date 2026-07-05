#!/bin/bash
cd /home/user/modded-nanogpt/experiments/cpu_micro
until grep -q ALL_DONE runner.log 2>/dev/null; do sleep 10; done
BESTLR=$(python3 -c "
import json
f = {}
for l in open('results.jsonl'):
    r = json.loads(l)
    if r['run'] == 'modern-muon' and 'final_val_loss' in r:
        f[r['lr']] = r['final_val_loss']
print(min(f, key=f.get))
")
echo "followups at muon lr=$BESTLR"
python3 train_micro.py --variant modern-muon --lr $BESTLR --budget 4000000 --ramp 64  --tag ramp64
python3 train_micro.py --variant modern-muon --lr $BESTLR --budget 4000000 --ema 0.99 --tag ema
echo ORIGINAL_DONE
