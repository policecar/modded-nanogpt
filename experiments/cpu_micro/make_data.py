"""Build the byte-level dataset for the CPU micro experiments.

Real English text from NLTK's gutenberg, webtext, and reuters corpora
(hosted on raw.githubusercontent.com via nltk_data), utf-8 bytes as tokens,
0x00 as document separator.

v2 design note: the corpus is only ~130 documents, several of them
multi-megabyte books, so a sequentially-read concatenated stream is highly
non-stationary (the model can spend the last quarter of training inside a
single book) and a contiguous validation slice is dominated by a handful of
documents. v1 of this experiment showed exactly that artifact: val loss on a
fixed eval set *rose* late in training for every variant as the training
stream drifted between styles. To make the stream stationary and the val set
representative, we cut the whole corpus into non-overlapping 513-byte windows
(512 inputs + 1 shifted target), shuffle them with a fixed seed, and hold out
5% as validation. Same data, same order, for every variant.

Usage: python3 make_data.py   (writes train_seqs.npy / val_seqs.npy here)
"""
import random
import numpy as np
import nltk

for c in ("gutenberg", "webtext", "reuters"):
    nltk.download(c, quiet=True)
from nltk.corpus import gutenberg, reuters, webtext

docs = []
for corpus in (gutenberg, webtext):
    for fid in corpus.fileids():
        docs.append(corpus.raw(fid))
rfids = reuters.fileids()  # thousands of tiny docs: group into chunks of 100
for i in range(0, len(rfids), 100):
    docs.append("\n\n".join(reuters.raw(f) for f in rfids[i:i + 100]))
print("docs:", len(docs), "total chars:", sum(len(d) for d in docs))

rng = random.Random(1234)
rng.shuffle(docs)
SEP = b"\x00"  # document boundary byte (analogous to <|endoftext|>)
blob = SEP.join(d.encode("utf-8", errors="replace") for d in docs) + SEP
arr = np.frombuffer(blob, dtype=np.uint8)

W = 513  # 512 input bytes + 1 for the shifted target
n = len(arr) // W
seqs = arr[: n * W].reshape(n, W).copy()
perm = np.random.default_rng(1234).permutation(n)
seqs = seqs[perm]
n_val = round(0.05 * n)
np.save("val_seqs.npy", seqs[:n_val])
np.save("train_seqs.npy", seqs[n_val:])
print(f"windows: {n} of {W} bytes; val {n_val} ({n_val*W/1e6:.2f}M tokens), "
      f"train {n - n_val} ({(n-n_val)*W/1e6:.2f}M tokens)")
