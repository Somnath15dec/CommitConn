"""Trie-constrained beam search over Semantic IDs.

Prompt format (shared with training): "Query: {q}\\nSID:" followed by 4 SID tokens <a_i><b_j><c_k><d_l>.
prefix_allowed_tokens_fn only allows SID-token continuations that exist in the catalog trie, so every finished beam
decodes to a catalog track. `exclude` removes track_ids (e.g. tune-half tracks for the Eval B pool) by pruning
the trie before search, so no beam is wasted on excluded tracks.
"""
from __future__ import annotations

import pandas as pd
import torch

LEVELS = ("a", "b", "c", "d")
PROMPT = "Query: {q}\nSID:"


def sid_token(level: str, i: int) -> str:
    return f"<{level}_{i}>"


def sid_token_list(sids: pd.DataFrame, k: int = 256) -> list[str]:
    """All SID tokens: a/b/c 0..k-1, d 0..max_d."""
    return ([sid_token(lv, i) for lv in "abc" for i in range(k)] +
            [sid_token("d", i) for i in range(int(sids.d.max()) + 1)])


def build_token_trie(sids: pd.DataFrame, tok_id: dict[str, int], exclude: set[str] | None = None) -> dict:
    """{a_id: {b_id: {c_id: {d_id: track_id}}}} over catalog SIDs, minus excluded track_ids."""
    trie: dict = {}
    for t, a, b, c, d in sids[["track_id", *LEVELS]].itertuples(index=False):
        if exclude and t in exclude:
            continue
        ids = [tok_id[sid_token(lv, int(v))] for lv, v in zip(LEVELS, (a, b, c, d))]
        trie.setdefault(ids[0], {}).setdefault(ids[1], {}).setdefault(ids[2], {})[ids[3]] = t
    return trie


class SidDecoder:
    def __init__(self, model, tokenizer, sids: pd.DataFrame, exclude: set[str] | None = None):
        self.model, self.tok = model, tokenizer
        self.tok_id = {t: tokenizer.convert_tokens_to_ids(t) for t in sid_token_list(sids)}
        self.trie = build_token_trie(sids, self.tok_id, exclude)
        self.fallback = [tokenizer.eos_token_id]

    def _allowed(self, generated: list[int]) -> list[int]:
        node = self.trie
        for t in generated:
            node = node.get(t) if isinstance(node, dict) else None
            if node is None:
                return self.fallback
        return list(node) if isinstance(node, dict) else self.fallback

    def lookup(self, ids: list[int]) -> str | None:
        node = self.trie
        for t in ids:
            if not isinstance(node, dict) or t not in node:
                return None
            node = node[t]
        return node if isinstance(node, str) else None

    @torch.no_grad()
    def search(self, queries: list[str], n: int = 10, num_beams: int = 20,
               batch_size: int = 16) -> list[list[tuple[str, float]]]:
        """Up to n distinct (track_id, score) per query; score = length-normalized beam log-prob."""
        num_beams = max(num_beams, n)
        self.tok.padding_side = "left"
        out = []
        for s in range(0, len(queries), batch_size):
            qs = queries[s:s + batch_size]
            enc = self.tok([PROMPT.format(q=q) for q in qs], return_tensors="pt", padding=True).to(
                self.model.device)
            plen = enc.input_ids.shape[1]
            gen = self.model.generate(
                **enc, max_new_tokens=len(LEVELS), num_beams=num_beams, num_return_sequences=num_beams,
                do_sample=False, early_stopping=True, output_scores=True, return_dict_in_generate=True,
                prefix_allowed_tokens_fn=lambda _b, ids: self._allowed(ids[plen:].tolist()),
                pad_token_id=self.tok.pad_token_id, eos_token_id=None)
            seqs = gen.sequences[:, plen:].tolist()
            scores = gen.sequences_scores.tolist()
            for qi in range(len(qs)):
                res, seen = [], set()
                for j in range(qi * num_beams, (qi + 1) * num_beams):
                    t = self.lookup(seqs[j][:len(LEVELS)])
                    if t is not None and t not in seen:
                        seen.add(t)
                        res.append((t, scores[j]))
                out.append(res[:n])
        return out
