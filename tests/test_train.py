"""CPU tests for train_data / train / decode, using a tiny random Qwen3 with the real Qwen3 tokenizer."""
import numpy as np
import pandas as pd
import pytest
import torch

from src import train as T
from src import train_data as D
from src.decode import LEVELS, SidDecoder, sid_token_list
from src.paraphrase import normalize


@pytest.fixture(scope="module")
def sids():
    return pd.read_parquet(D.SIDS)


@pytest.fixture(scope="module")
def tiny(sids):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    tok = T.load_tokenizer()
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                      max_position_embeddings=256)
    torch.manual_seed(0)
    model, new_ids = T.build_model(tok, sids, base=Qwen3ForCausalLM(cfg))
    model.eval()
    return model, tok, new_ids


@pytest.fixture(scope="module")
def pool():
    return D.load_pool()[0]


# ---------------------------------------------------------------- tokens
def test_token_count_matches_trie(tiny, sids):
    _, tok, new_ids = tiny
    expected = 3 * 256 + int(sids.d.max()) + 1
    assert len(new_ids) == len(set(new_ids)) == expected == len(sid_token_list(sids))
    assert len(tok) == new_ids[-1] + 1
    for lv in LEVELS:                                   # every digit used in the trie has its own token
        used = set(sids[lv].astype(int))
        assert all(tok.convert_tokens_to_ids(f"<{lv}_{i}>") in new_ids for i in used)
    assert tok.convert_ids_to_tokens(T.sid_ids(tok, (3, 7, 0, 2))) == ["<a_3>", "<b_7>", "<c_0>", "<d_2>"]


def test_only_lora_and_new_token_rows_train(tiny):
    model, _, new_ids = tiny
    extra = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    assert len(extra) == 1 and "trainable_tokens_delta" in extra[0]
    delta = dict(model.named_parameters())[extra[0]]
    assert delta.shape[0] == len(new_ids)
    assert type(model.base_model.model.lm_head).__name__ == "TrainableTokensWrapper"     # tied head shares delta


# ---------------------------------------------------------------- loss mask
def test_loss_mask_covers_only_sid_tokens(tiny):
    _, tok, _ = tiny
    ids, labels = T.encode(tok, "calm music for a rainy day", (1, 2, 3, 0))
    sid = T.sid_ids(tok, (1, 2, 3, 0))
    kept = [(i, l) for i, l in zip(ids, labels) if l != -100]
    assert [l for _, l in kept] == sid and [i for i, _ in kept] == sid
    assert ids[-1] == tok.eos_token_id and labels[-1] == -100
    assert len(ids) <= T.MAX_LEN
    long_ids, long_lab = T.encode(tok, "word " * 200, (1, 2, 3, 0))
    assert len(long_ids) == T.MAX_LEN and long_lab[-5:-1] == sid


def test_collate_left_pads_and_loss_targets_are_labels(tiny):
    model, tok, _ = tiny
    b = T.collate([T.encode(tok, "gym", (4, 5, 6, 1)), T.encode(tok, "music for a long drive home", (7, 8, 9, 0))],
                  tok.pad_token_id)
    lab = b["labels"]
    assert ((lab != -100).sum(1) == 4).all()
    assert (lab[:, -5:-1] != -100).all() and (lab[:, :-5] == -100).all() and (lab[:, -1] == -100).all()
    assert b["attention_mask"][0, 0] == 0 and b["position_ids"][:, -1].tolist() == (b["attention_mask"].sum(1) - 1).tolist()
    loss = T.sid_loss(model, b)
    # reference: full-logit cross-entropy on the labelled positions only
    full = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=b["position_ids"]).logits
    ref = torch.nn.functional.cross_entropy(full[:, :-1].float().reshape(-1, full.size(-1)), lab[:, 1:].reshape(-1),
                                            ignore_index=-100)
    assert torch.allclose(loss, ref, atol=1e-4)


# ---------------------------------------------------------------- decoding
def test_untrained_decode_returns_valid_catalog_tracks(tiny, sids):
    model, tok, _ = tiny
    dec = SidDecoder(model, tok, sids)
    res = dec.search(["calm piano for sleep", "gym rap"], n=10, num_beams=10)
    cat = set(sids.track_id)
    for r in res:
        ids = [t for t, _ in r]
        assert 0 < len(ids) <= 10 and len(set(ids)) == len(ids) and set(ids) <= cat
        assert all(np.isfinite(s) for _, s in r)


def test_exclude_option(tiny, sids):
    model, tok, _ = tiny
    first = [t for t, _ in SidDecoder(model, tok, sids).search(["calm piano for sleep"], n=5, num_beams=5)[0]]
    dec = SidDecoder(model, tok, sids, exclude=set(first))
    again = [t for t, _ in dec.search(["calm piano for sleep"], n=5, num_beams=5)[0]]
    assert again and not set(again) & set(first)
    keep = set(sids.track_id.iloc[:30])
    only = SidDecoder(model, tok, sids, exclude=set(sids.track_id) - keep)
    got = [t for t, _ in only.search(["anything"], n=10, num_beams=10)[0]]
    assert got and set(got) <= keep


def test_allowed_tokens_follow_trie(tiny, sids):
    model, tok, _ = tiny
    dec = SidDecoder(model, tok, sids)
    r = sids.iloc[0]
    path = T.sid_ids(tok, tuple(int(r[l]) for l in LEVELS))
    for k in range(4):
        assert path[k] in dec._allowed(path[:k])
    assert dec.lookup(path) == r.track_id
    assert dec._allowed([tok.eos_token_id]) == [tok.eos_token_id]


# ---------------------------------------------------------------- pool
def test_pool_has_no_eval_strings_and_no_eval_half_only_human_pairs(pool):
    ev = D.eval_strings()
    assert not set(pool["query"].map(normalize)) & ev
    eval_only = D.eval_half_only_tracks()
    human = pool[pool.source == "tune_context"]
    assert len(human) and not set(human.track_id) & eval_only


def test_val_split_and_epoch_sampler(pool):
    tr, val = D.split_val(pool)
    assert len(val) == D.VAL_N and len(tr) + len(val) == len(pool)
    key = lambda df: set(zip(df["query"], df.track_id))
    assert not key(tr) & key(val)
    tr2, val2 = D.split_val(pool)
    assert key(val) == key(val2)
    e0, e0b, e1 = D.epoch_indices(tr, 0), D.epoch_indices(tr, 0), D.epoch_indices(tr, 1)
    assert (e0 == e0b).all() and not np.array_equal(np.sort(e0), np.sort(e1))
    assert tr.iloc[e0]["query"].value_counts().max() <= D.MAX_PER_STRING
    small = tr["query"].map(tr["query"].value_counts()) <= D.MAX_PER_STRING
    assert set(np.flatnonzero(small.to_numpy())) <= set(e0)              # strings under the cap are kept whole


def test_warmup_cosine_schedule():
    w = T.WARMUP_STEPS
    assert w == 30
    assert T.cosine_lr(0, 130) == pytest.approx(1 / w) and T.cosine_lr(w - 1, 130) == pytest.approx(1.0)
    assert T.cosine_lr(w, 130) == pytest.approx(1.0)
    assert T.cosine_lr(w + 50, 130) == pytest.approx(0.5)
    assert T.cosine_lr(130, 130) == pytest.approx(0.0, abs=1e-9)


def test_best_epoch_rule():
    t = [{"epoch": 1, "dev_b_precision@10": 0.3, "val_loss": 2.0}, {"epoch": 2, "dev_b_precision@10": 0.4, "val_loss": 1.5},
         {"epoch": 3, "dev_b_precision@10": 0.4, "val_loss": 1.2}]
    assert T.best_epoch(t)["epoch"] == 3
    assert T.best_epoch(t[:2])["epoch"] == 2
