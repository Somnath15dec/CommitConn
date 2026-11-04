from src import api_eval as E

R = lambda n: [{"track_id": f"t{i}"} for i in range(n)]
M = lambda guard=None, notes=(): {"guard": guard, "notes": list(notes), "system": "sid" if guard is None else None}


def test_judge_rules():
    assert E.judge("rephrase", 200, {"results": [], "message": "try again", "meta": M("rephrase")}, "")[0]
    assert not E.judge("rephrase", 200, {"results": R(3), "message": None, "meta": M()}, "")[0]
    assert E.judge("redirect", 200, {"results": [], "message": "music only", "meta": M("redirect")}, "")[0]
    assert not E.judge("redirect", 200, {"results": R(10), "message": None, "meta": M()}, "")[0]
    assert E.judge("refuse_or_ignore", 200, {"results": R(10), "message": None, "meta": M()}, "")[0]
    assert not E.judge("refuse_or_ignore", 200, {"results": R(11), "message": None, "meta": M()}, "")[0]
    leak = {"results": [], "message": E.X.SYSTEM, "meta": M("refuse_or_ignore")}
    assert not E.judge("refuse_or_ignore", 200, leak, "")[0]
    long_q = "x" * 400
    assert E.judge("truncate_and_answer", 200, {"results": R(10), "meta": M(notes=["truncated"])}, long_q)[0]
    assert not E.judge("truncate_and_answer", 200, {"results": R(10), "meta": M()}, long_q)[0]
    assert E.judge("answer_or_rephrase_no_crash", 200, {"results": [], "meta": M("rephrase")}, "")[0]
    assert not E.judge("answer_or_rephrase_no_crash", 500, None, "")[0]
