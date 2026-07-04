# Sample efficiency of this fork

**Goal:** make the model a lot more sample efficient without ceding performance
(≤ 3.28 validation loss on FineWeb, ≤ 124M active parameters, unmodified train/val
data pipelines — i.e. the NanoGPT speedrun rules, no cheating).

**How it was achieved:** this fork was previously frozen at the November 2024 state
of the speedrun (`train_gpt2.py`, records 1–12). It has now been synced with
[upstream](https://github.com/KellerJordan/modded-nanogpt) master, which carries the
current world-record training algorithm (`train_gpt.py`, 84 records as of May 2026).
Every record is validated by the community against the same fixed benchmark: the
same FineWeb `.bin` shards, the same fixed 10,485,760-token validation set, and a
statistically significant mean validation loss ≤ 3.28 over multiple runs.

## Token budget: before vs. after

| | steps | tokens/step | total training tokens | val loss |
|---|---|---|---|---|
| llm.c GPT-2 baseline (05/2024) | 19,560 | 524,288 | 10.0B | ~3.29 |
| this fork before sync (11/2024 state) | 3,242 | 524,288 | 1.70B | ~3.278 |
| this fork after sync (05/2026 record) | 1,390 | 131,072 → 393,216 (batch-size schedule) | **0.366B** | ≤ 3.28 |

The current trainer's batch-size schedule (8→16→24 sequences of ≤2048 tokens × 8 GPUs,
with 1380 scheduled steps in three equal stages plus 10 extension steps) consumes
460 × 131,072 + 460 × 262,144 + 460 × 393,216 + 10 × 393,216 = **365,690,880 tokens**.

That is a **4.6× sample-efficiency improvement** over this fork's previous state, and
**27× over the original llm.c GPT-2 replication**, at identical validation performance.

## Which changes drive the sample efficiency

Most post-2024 records improved wall-clock speed (kernels, FP8, comms overlap), but the
token-budget reduction from 1.7B → 0.37B comes from algorithmic changes, including:

* FlexAttention with document-aligned block-causal masking, long-short sliding-window
  attention, and window-size warmup (+ YaRN when windows grow)
* Learnable value embeddings mixed into attention values (extending value-residual
  learning), later gated and sparsified
* U-net-style and MUDD skip connections with learnable weights
* Logit softcapping at 15 with asymmetric rescale
* Muon refinements: Polar Express orthogonalization, NorMuon variance normalization,
  cautious weight decay tied to the LR schedule
* Batch-size and max-sequence-length schedules (critical-batch-size literature)
* Multi-token prediction with a decaying auxiliary weight; embed/lm_head tied for the
  first 2/3 of training
* Smear module (1-token lookback), bigram hash embeddings, paired-head attention,
  partial key offset

See the record history in [README.md](README.md) and per-record writeups with
reproducible logs under [records/track_1_short/](records/track_1_short/) for
attribution and ablations. There is also an
[optimization track](records/track_3_optimization) dedicated specifically to
minimizing training steps under a fixed architecture/data/batch-size budget.
