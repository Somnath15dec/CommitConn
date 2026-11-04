import pandas as pd
import pytest
import torch

from src import dial as Dl
from src import train as T
from src.decode import SidDecoder
from src.paths import CATALOG, REAL_USERS
from src.train_data import SIDS


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    tok = T.load_tokenizer()
    sids = pd.read_parquet(SIDS)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                      max_position_embeddings=256)
    torch.manual_seed(0)
    model, _ = T.build_model(tok, sids, base=Qwen3ForCausalLM(cfg))
    return model.eval(), tok, sids


QUERIES = ["music for the gym", "a playlist for sleep"]


def test_lambda_zero_reproduces_normal_decoding(tiny):
    model, tok, sids = tiny
    dec = SidDecoder(model, tok, sids)
    H = set(sids.a.astype(int).iloc[:500])
    plain = dec.search(QUERIES, n=10, num_beams=10, batch_size=2)
    zero = dec.search(QUERIES, n=10, num_beams=10, batch_size=2, logits_processor_fn=Dl.dial_fn(tok, H, 0))
    assert plain == zero


def test_large_negative_lambda_avoids_history_codes(tiny):
    model, tok, sids = tiny
    dec = SidDecoder(model, tok, sids)
    level1 = dict(zip(sids.track_id, sids.a.astype(int)))
    plain = dec.search(QUERIES, n=10, num_beams=10, batch_size=2)
    H = {level1[t] for r in plain for t, _ in r}                      # the codes it would normally pick
    assert len(H) < 256                                               # alternatives exist
    neg = dec.search(QUERIES, n=10, num_beams=10, batch_size=2, logits_processor_fn=Dl.dial_fn(tok, H, -1e4))
    assert all(r for r in neg) and all(level1[t] not in H for r in neg for t, _ in r)
    pos = dec.search(QUERIES, n=10, num_beams=10, batch_size=2,
                     logits_processor_fn=Dl.dial_fn(tok, {level1[t] for t, _ in plain[0][-1:]}, 1e4))
    assert all(level1[t] == level1[plain[0][-1][0]] for t, _ in pos[0])  # strong positive -> that code only


def test_bias_only_on_first_step():
    p = Dl.FirstTokenBias(prompt_len=5, token_ids=[1, 3], lam=2.0)
    s = torch.zeros(2, 6)
    assert p(torch.zeros(2, 5, dtype=torch.long), s)[:, [1, 3]].eq(2).all()
    assert p(torch.zeros(2, 6, dtype=torch.long), s).eq(0).all()        # later steps untouched
    assert torch.isinf(p(torch.zeros(1, 5, dtype=torch.long), torch.full((1, 6), -float("inf")))).all()


def test_users_and_queries_follow_plan():
    cat = pd.read_parquet(CATALOG)
    users = Dl.select_users(pd.read_parquet(REAL_USERS), cat)
    assert len(users) == Dl.N_USERS and all(len(u["history"]) >= Dl.MIN_HISTORY for u in users)
    assert [u["user_id"] for u in Dl.select_users(pd.read_parquet(REAL_USERS), cat)] == [u["user_id"] for u in users]
    qs = Dl.user_queries(users)
    assert len(qs) == 2 * len(users)
    by = {u["user_id"]: u for u in users}
    for uid in by:
        mine = [q for q in qs if q["user_id"] == uid]
        assert len({q["context_word"] for q in mine}) == 2
        assert all(q["context_word"] in by[uid]["context_words"] for q in mine)
        assert all(q["template_id"] in Dl.CONTEXT_TEMPLATE_IDS for q in mine)


def test_metrics_and_criteria():
    level1 = {"a": 1, "b": 2, "c": 3}
    assert Dl.novelty(["a", "b", "c"], level1, {1}) == pytest.approx(0.2)
    assert Dl.familiar_hit(["a", "b"], {"b"}) == pytest.approx(0.1)
    assert Dl.relevance(["a", "b"], {"a": ["gym"], "b": []}, "gym") == pytest.approx(0.1)
    mk = lambda n, f, r: {"novelty@10": {"mean": n}, "familiar_hit@10": {"mean": f}, "relevance@10": {"mean": r}}
    t = {4: mk(.1, .5, .5), 2: mk(.2, .4, .5), 0: mk(.3, .3, .5), -2: mk(.4, .2, .45), -4: mk(.5, .1, .4)}
    c = Dl.criteria(t)
    assert c["novelty_strictly_increases_+4_to_-4"] and c["relevance_at_-4_ge_70pct_of_0"] and c["familiar_hit_highest_at_+4"]
    t[-4] = mk(.5, .1, .3)
    assert not Dl.criteria(t)["relevance_at_-4_ge_70pct_of_0"]
