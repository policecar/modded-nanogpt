"""Build the byte-level dataset for the CPU micro experiments.

Real English text from NLTK's gutenberg, webtext, and reuters corpora
(hosted on raw.githubusercontent.com via nltk_data), document-shuffled with a
fixed seed, utf-8 bytes as tokens, 0x00 as document separator. First 1M bytes
are held out as the fixed validation set.

Usage: python3 make_data.py   (writes train_tokens.npy / val_tokens.npy here)
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
VAL = 1_048_576
np.save("val_tokens.npy", arr[:VAL])
np.save("train_tokens.npy", arr[VAL:])
print("val:", VAL, "train:", len(arr) - VAL)
