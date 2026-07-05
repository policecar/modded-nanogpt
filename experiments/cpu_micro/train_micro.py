"""
CPU micro-scale sample-efficiency experiments.

Byte-level GPT (~1.2M non-embedding params) trained on real English text
(NLTK gutenberg+webtext+reuters, document-shuffled, byte tokens, 0x00 as doc
separator). All variants see the exact same token stream and are evaluated on
the same fixed held-out validation set at a fixed training-token budget, so
differences in the loss-vs-tokens curve measure sample efficiency.

Variants decompose the modded-nanogpt recipe:
  A. gpt2-adamw   : GPT-2-style arch (learned pos emb, LayerNorm, GELU, tied
                    embed/head, std init) + AdamW
  B. modern-adamw : modernized arch (rotary, RMSNorm, QK-norm, ReLU^2,
                    zero-init projections, untied zero-init head, logit
                    softcap) + AdamW
  C. modern-muon  : same arch as B, hidden matrices trained with Muon
                    (Newton-Schulz orthogonalized momentum) + Adam for
                    embed/head/scalars, momentum warmup
  D. shortcut-muon: C + value embeddings, embedding shortcut (x0 lambdas) and
                    U-net skip weights

This obviously cannot certify the 124M/3.28 FineWeb result (that needs 8xH100);
it tests, at CPU scale, whether the recipe's components actually buy sample
efficiency compared to the GPT-2 baseline under a controlled budget.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(os.cpu_count())
HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- data

def load_seqs(name):
    # (N, 513) uint8 windows: 512 input bytes + 1 for the shifted target,
    # pre-shuffled with a fixed seed by make_data.py (see v2 design note there)
    return torch.from_numpy(np.load(os.path.join(HERE, name)).astype(np.int64))

class Stream:
    """Deterministic batch stream over pre-shuffled windows (identical for every run)."""
    def __init__(self, seqs, B, T):
        assert seqs.size(1) == T + 1
        self.seqs, self.B, self.pos = seqs, B, 0
    def next(self):
        if self.pos + self.B > len(self.seqs):
            self.pos = 0
        buf = self.seqs[self.pos : self.pos + self.B]
        self.pos += self.B
        return buf[:, :-1], buf[:, 1:]

# ----------------------------------------------------------------- model

def rmsnorm(x):
    return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    # half-truncated rotary as in the speedrun (base freq 1024)
    def __init__(self, head_dim, max_len):
        super().__init__()
        inv = (1 / 1024) ** torch.linspace(0, 1, steps=head_dim // 4)
        inv = torch.cat([inv, torch.zeros(head_dim // 4)])
        theta = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x):  # (B, T, H, D)
        T = x.size(1)
        cos, sin = self.cos[None, :T, None, :], self.sin[None, :T, None, :]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, self.h = cfg.dim, cfg.heads
        self.hd = d // self.h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.cfg = cfg
        self.rotary = Rotary(self.hd, cfg.seq_len) if cfg.rotary else None
        if cfg.shortcuts:
            self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x, ve=None):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3 * self.h, self.hd).chunk(3, dim=-2)
        if self.cfg.qk_norm:
            q, k = rmsnorm(q), rmsnorm(k)
        if self.rotary is not None:
            q, k = self.rotary(q), self.rotary(k)
        if self.cfg.shortcuts:
            v = self.lambdas[0] * v + (self.lambdas[1] * ve.view_as(v) if ve is not None else 0)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, d))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.dim, 4 * cfg.dim, bias=False)
        self.proj = nn.Linear(4 * cfg.dim, cfg.dim, bias=False)
        self.relu2 = cfg.relu2

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x).square() if self.relu2 else F.gelu(x)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att, self.mlp, self.cfg = Attention(cfg), MLP(cfg), cfg
        if cfg.rmsnorm:
            self.n1 = self.n2 = rmsnorm
        else:
            self.ln1, self.ln2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)
            self.n1, self.n2 = self.ln1, self.ln2
        if cfg.shortcuts:
            self.x0_lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x, x0=None, ve=None):
        if self.cfg.shortcuts:
            x = self.x0_lambdas[0] * x + self.x0_lambdas[1] * x0
        x = x + self.att(self.n1(x), ve)
        x = x + self.mlp(self.n2(x))
        return x

class Config:
    def __init__(self, **kw):
        self.vocab, self.dim, self.heads, self.layers, self.seq_len = 256, 128, 4, 6, 512
        self.rotary = self.rmsnorm = self.qk_norm = self.relu2 = False
        self.zero_init = self.untied_head = self.shortcuts = False
        self.softcap = None
        self.__dict__.update(kw)

class MicroGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.dim)
        self.wpe = None if cfg.rotary else nn.Embedding(cfg.seq_len, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.lnf = (lambda x: rmsnorm(x)) if cfg.rmsnorm else nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        if cfg.shortcuts:
            n_enc = cfg.layers // 2
            self.skip_w = nn.Parameter(torch.ones(cfg.layers - n_enc))
            self.ve = nn.ModuleList(nn.Embedding(cfg.vocab, cfg.dim) for _ in range(2))
        self.reset_parameters()
        if not cfg.untied_head:
            self.head.weight = self.wte.weight

    def reset_parameters(self):
        cfg = self.cfg
        if cfg.zero_init:  # speedrun-style init
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    std = 0.5 * m.in_features ** -0.5
                    nn.init.uniform_(m.weight, -(3 ** 0.5) * std, (3 ** 0.5) * std)
            for b in self.blocks:
                b.att.proj.weight.detach().zero_()
                b.mlp.proj.weight.detach().zero_()
            self.head.weight.detach().zero_()
            nn.init.normal_(self.wte.weight, std=0.02)
        else:  # GPT-2-style init
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(m.weight, std=0.02)
            for b in self.blocks:  # scaled residual projections
                nn.init.normal_(b.att.proj.weight, std=0.02 / math.sqrt(2 * cfg.layers))
                nn.init.normal_(b.mlp.proj.weight, std=0.02 / math.sqrt(2 * cfg.layers))

    def forward(self, idx, targets, loss_w=None):
        B, T = idx.shape
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(T, device=idx.device))[None]
        if self.cfg.rmsnorm:
            x = rmsnorm(x)
        if self.cfg.shortcuts:
            x0 = x
            ve = [self.ve[0](idx), self.ve[1](idx)] + [None] * (self.cfg.layers - 4) \
                 + [self.ve[0](idx), self.ve[1](idx)]
            n_enc = self.cfg.layers // 2
            skips = []
            for i in range(n_enc):
                x = self.blocks[i](x, x0, ve[i])
                skips.append(x)
            for i in range(n_enc, self.cfg.layers):
                x = x + self.skip_w[i - n_enc] * skips.pop()
                x = self.blocks[i](x, x0, ve[i])
        else:
            for b in self.blocks:
                x = b(x)
        logits = self.head(self.lnf(x))
        if self.cfg.softcap:
            c = self.cfg.softcap
            logits = c * torch.tanh(logits / c)
        if loss_w is None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        # position-weighted CE (training only): downweight context-poor early positions
        lo = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                             reduction="none").view(B, T)
        return (lo * loss_w).sum() / (B * loss_w.sum())

# ----------------------------------------------------------------- muon

def ns5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.05, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(p)
                buf = st["mom"]
                buf.lerp_(p.grad, 1 - g["momentum"])
                u = ns5(p.grad.lerp(buf, g["momentum"]))
                p.add_(u, alpha=-g["lr"] * max(1, p.size(0) / p.size(1)) ** 0.5)

# ----------------------------------------------------------------- training

def evaluate(model, val_tokens, B, T, max_tokens):
    model.eval()
    s, losses = Stream(val_tokens, B, T), []
    with torch.no_grad():
        for _ in range(max_tokens // (B * T)):
            x, y = s.next()
            losses.append(model(x, y).item())
    model.train()
    return sum(losses) / len(losses)

def run(name, cfg, opt_name, lr, budget, out, B=16, T=512, evals=8, final_eval_tokens=524288,
        ramp=0, ema=0.0):
    # ramp>0: linearly ramp per-position CE weight over the first `ramp` positions
    #         during training (eval is always unweighted).
    # ema>0:  maintain a Polyak average of weights with this decay; report val loss
    #         of both raw and averaged weights.
    torch.manual_seed(42)
    train_tokens, val_tokens = load_seqs("train_seqs.npy"), load_seqs("val_seqs.npy")
    model = MicroGPT(cfg)
    loss_w = torch.clamp(torch.arange(1, T + 1).float() / ramp, max=1.0) if ramp else None
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if ema else None
    nparams = sum(p.numel() for p in model.parameters())
    steps = budget // (B * T)
    print(f"[{name} lr={lr}] params={nparams} steps={steps}", flush=True)

    if opt_name == "adamw":
        decay = [p for p in model.parameters() if p.ndim >= 2]
        other = [p for p in model.parameters() if p.ndim < 2]
        opts = [torch.optim.AdamW([dict(params=decay, weight_decay=0.1),
                                   dict(params=other, weight_decay=0.0)],
                                  lr=lr, betas=(0.9, 0.95), eps=1e-8)]
        warmup = max(10, int(0.05 * steps))
    else:  # muon on hidden matrices, adam on the rest
        hidden = [p for n, p in model.named_parameters()
                  if p.ndim == 2 and not any(k in n for k in ("wte", "wpe", "head", "ve"))]
        rest_embed = [p for n, p in model.named_parameters()
                      if any(k in n for k in ("wte", "wpe", "ve"))]
        rest_head = [p for n, p in model.named_parameters() if "head" in n and p.ndim == 2]
        scalars = [p for p in model.parameters() if p.ndim < 2]
        muon = Muon(hidden, lr=lr, momentum=0.95)
        adam = torch.optim.Adam([dict(params=rest_embed, lr=3e-3),
                                 dict(params=rest_head, lr=3e-3),
                                 dict(params=scalars, lr=1e-2)],
                                betas=(0.8, 0.95), eps=1e-10)
        opts = [muon, adam]
        warmup = 0

    def lr_mul(step):  # linear warmup -> constant -> linear cooldown to 0.1x (last 40%)
        if step < warmup:
            return (step + 1) / warmup
        t = 1 - step / steps
        w = min(t / 0.4, 1.0)
        return w + (1 - w) * 0.1

    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_mul) for o in opts]
    stream = Stream(train_tokens, B, T)
    eval_every = max(1, steps // evals)
    t0 = time.time()
    for step in range(steps):
        x, y = stream.next()
        loss = model(x, y, loss_w)
        loss.backward()
        if opt_name == "muon":  # momentum warmup over first 20% of training
            frac = min(step / max(1, int(0.2 * steps)), 1.0)
            for grp in opts[0].param_groups:
                grp["momentum"] = 0.85 * (1 - frac) + 0.95 * frac
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        model.zero_grad(set_to_none=True)
        if ema_state is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_state[k].lerp_(v.float(), 1 - ema) if v.is_floating_point() else ema_state[k].copy_(v)
        if (step + 1) % eval_every == 0 or step == steps - 1:
            vl = evaluate(model, val_tokens, B, T, 131072)
            rec = dict(run=name, lr=lr, step=step + 1, tokens=(step + 1) * B * T,
                       train_loss=round(loss.item(), 4), val_loss=round(vl, 4),
                       secs=round(time.time() - t0, 1))
            if ema_state is not None:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state)
                rec["val_loss_ema"] = round(evaluate(model, val_tokens, B, T, 131072), 4)
                model.load_state_dict(backup)
            print(json.dumps(rec), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    final = evaluate(model, val_tokens, B, T, final_eval_tokens)
    rec = dict(run=name, lr=lr, final_val_loss=round(final, 4), params=nparams,
               budget=budget, secs=round(time.time() - t0, 1))
    if ema_state is not None:
        model.load_state_dict(ema_state)
        rec["final_val_loss_ema"] = round(evaluate(model, val_tokens, B, T, final_eval_tokens), 4)
    print(json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return final

VARIANTS = {
    "gpt2-adamw":    (Config(), "adamw"),
    "modern-adamw":  (Config(rotary=True, rmsnorm=True, qk_norm=True, relu2=True,
                             zero_init=True, untied_head=True, softcap=15), "adamw"),
    "modern-muon":   (Config(rotary=True, rmsnorm=True, qk_norm=True, relu2=True,
                             zero_init=True, untied_head=True, softcap=15), "muon"),
    "shortcut-muon": (Config(rotary=True, rmsnorm=True, qk_norm=True, relu2=True,
                             zero_init=True, untied_head=True, softcap=15,
                             shortcuts=True), "muon"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--budget", type=int, default=2_000_000)
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--ramp", type=int, default=0, help="ramp CE weight over first N positions (train only)")
    ap.add_argument("--ema", type=float, default=0.0, help="Polyak weight-averaging decay (0=off)")
    ap.add_argument("--tag", default="", help="suffix for the run name")
    a = ap.parse_args()
    cfg, opt = VARIANTS[a.variant]
    if a.bench:
        torch.manual_seed(42)
        tr = load_seqs("train_seqs.npy")
        m, s = MicroGPT(cfg), Stream(tr, 16, 512)
        for _ in range(2):  # warm
            x, y = s.next(); m(x, y).backward(); m.zero_grad(set_to_none=True)
        t = time.time(); n = 5
        for _ in range(n):
            x, y = s.next(); m(x, y).backward(); m.zero_grad(set_to_none=True)
        dt = (time.time() - t) / n
        print(f"{a.variant}: {dt:.2f}s/step, {16*512/dt:.0f} tok/s (fwd+bwd)")
    else:
        name = a.variant + (f"+{a.tag}" if a.tag else "")
        run(name, cfg, opt, a.lr, a.budget, a.out, ramp=a.ramp, ema=a.ema)
