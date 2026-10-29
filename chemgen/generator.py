"""Property-conditioned SMILES language model.

A GRU reads SMILES token by token. The condition vector (target logP, QED and
synthetic accessibility) initialises the hidden state and is also fed in at every
step, so sampling with a chosen condition steers generation toward those values.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from . import chem
from .tokenizer import Vocab, tokenize

CONDITION_COLUMNS = ["logP", "QED", "SAS"]
MAX_TOKENS = 80


class ConditionalSmilesLM(nn.Module):
    def __init__(self, vocab_size: int, cond_dim: int, emb: int = 128, hidden: int = 384,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, cond_dim=cond_dim, emb=emb, hidden=hidden,
                           layers=layers, dropout=dropout)
        self.layers, self.hidden = layers, hidden
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.cond_to_h = nn.Linear(cond_dim, layers * hidden)
        self.gru = nn.GRU(emb + cond_dim, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def _init_hidden(self, cond: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.cond_to_h(cond))
        return h.view(-1, self.layers, self.hidden).transpose(0, 1).contiguous()

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor, h: torch.Tensor | None = None):
        x = torch.cat([self.embed(tokens), cond.unsqueeze(1).expand(-1, tokens.size(1), -1)], dim=-1)
        out, h = self.gru(x, self._init_hidden(cond) if h is None else h)
        return self.out(self.drop(out)), h

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, vocab: Vocab, max_len: int = MAX_TOKENS + 2,
               temperature: float = 1.0) -> list[list[int]]:
        self.eval()
        n = cond.size(0)
        tok = torch.full((n, 1), vocab.bos_id, dtype=torch.long)
        done = torch.zeros(n, dtype=torch.bool)
        pad = torch.full((n, 1), vocab.pad_id, dtype=torch.long)
        h, seqs = None, []
        for _ in range(max_len):
            logits, h = self(tok, cond, h)
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            nxt = torch.where(done.unsqueeze(1), pad, nxt)
            seqs.append(nxt)
            done |= nxt.squeeze(1) == vocab.eos_id
            tok = nxt
            if done.all():
                break
        return torch.cat(seqs, dim=1).tolist()


class Generator:
    """Model + vocabulary + condition normalisation, loadable from a single checkpoint."""

    def __init__(self, model: ConditionalSmilesLM, vocab: Vocab, cond_mean: np.ndarray,
                 cond_std: np.ndarray):
        self.model, self.vocab = model, vocab
        self.cond_mean, self.cond_std = cond_mean, cond_std

    def _normalise(self, targets: np.ndarray) -> torch.Tensor:
        return torch.tensor((targets - self.cond_mean) / self.cond_std, dtype=torch.float32)

    def sample(self, logp: float, qed: float, sas: float, n: int = 1000,
               temperature: float = 1.0, batch_size: int = 500) -> list[str]:
        targets = np.tile(np.array([logp, qed, sas], dtype=np.float32), (n, 1))
        smiles: list[str] = []
        for start in range(0, n, batch_size):
            cond = self._normalise(targets[start:start + batch_size])
            smiles += [self.vocab.decode(ids) for ids in self.model.sample(cond, self.vocab,
                                                                          temperature=temperature)]
        return smiles

    def save(self, path: Path) -> None:
        torch.save({"state": self.model.state_dict(), "config": self.model.config,
                    "vocab": self.vocab.itos, "cond_mean": self.cond_mean,
                    "cond_std": self.cond_std}, path)

    @classmethod
    def load(cls, path: Path) -> "Generator":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = ConditionalSmilesLM(**ckpt["config"])
        model.load_state_dict(ckpt["state"])
        model.eval()
        return cls(model, Vocab.from_list(ckpt["vocab"]), ckpt["cond_mean"], ckpt["cond_std"])


def _batches(encoded: list[list[int]], conds: np.ndarray, vocab: Vocab, batch_size: int,
             rng: np.random.Generator):
    order = rng.permutation(len(encoded))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        inputs = pad_sequence([torch.tensor([vocab.bos_id] + encoded[i]) for i in idx],
                              batch_first=True, padding_value=vocab.pad_id)
        targets = pad_sequence([torch.tensor(encoded[i] + [vocab.eos_id]) for i in idx],
                               batch_first=True, padding_value=vocab.pad_id)
        yield inputs, targets, torch.tensor(conds[idx], dtype=torch.float32)


def _validity(gen: Generator, conds_raw: np.ndarray, n: int = 300) -> float:
    rows = conds_raw[:n]
    cond = gen._normalise(rows)
    seqs = gen.model.sample(cond, gen.vocab)
    return float(np.mean([chem.to_mol(gen.vocab.decode(s)) is not None for s in seqs]))


def train_generator(csv_path: Path, out_path: Path, train_smiles_path: Path, n_mols: int = 100_000,
                    epochs: int = 12, batch_size: int = 256, lr: float = 2e-3, seed: int = 0,
                    log=print) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    df = pd.read_csv(csv_path).rename(columns={"qed": "QED"})
    df["smiles"] = df["smiles"].str.strip()
    df = df[df["smiles"].map(lambda s: len(tokenize(s)) <= MAX_TOKENS)]
    df = df.sample(n=min(n_mols, len(df)), random_state=seed).reset_index(drop=True)
    log(f"training on {len(df)} molecules")

    # Canonicalise once so novelty checks at generation time compare like with like.
    canon = [chem.canonical(s) for s in df["smiles"]]
    keep = [c is not None for c in canon]
    df, canon = df[keep].reset_index(drop=True), [c for c in canon if c is not None]
    train_smiles_path.write_text("\n".join(canon))

    vocab = Vocab.build(canon)
    encoded = [vocab.encode(s) for s in canon]
    raw = df[CONDITION_COLUMNS].to_numpy(dtype=np.float32)
    cond_mean, cond_std = raw.mean(0), raw.std(0)
    conds = (raw - cond_mean) / cond_std

    n_val = max(500, len(encoded) // 50)
    val_enc, val_cond, val_raw = encoded[:n_val], conds[:n_val], raw[:n_val]
    tr_enc, tr_cond = encoded[n_val:], conds[n_val:]

    model = ConditionalSmilesLM(len(vocab), len(CONDITION_COLUMNS))
    gen = Generator(model, vocab, cond_mean, cond_std)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * ((len(tr_enc) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    best, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, count = time.time(), 0.0, 0
        for inputs, targets, cond in _batches(tr_enc, tr_cond, vocab, batch_size, rng):
            logits, _ = model(inputs, cond)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, count = total + loss.item(), count + 1

        model.eval()
        with torch.no_grad():
            val = [loss_fn(model(i, c)[0].reshape(-1, len(vocab)), t.reshape(-1)).item()
                   for i, t, c in _batches(val_enc, val_cond, vocab, 512, rng)]
        val_loss = float(np.mean(val))
        validity = _validity(gen, val_raw)
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss,
                        "validity": validity})
        log(f"epoch {epoch:2d}/{epochs}  train {total / count:.3f}  val {val_loss:.3f}  "
            f"validity {validity:.1%}  ({time.time() - t0:.0f}s)")
        if val_loss < best:
            best = val_loss
            gen.save(out_path)
    return {"best_val_loss": best, "history": history}
