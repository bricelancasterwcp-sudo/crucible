from crucible.run.types import Candidate

def test_candidate_round_trips_through_json():
    import json
    c = Candidate("def f():\n    return 1\n", -0.42, 0.83)
    assert Candidate.from_dict(json.loads(json.dumps(c.to_dict()))) == c

def test_candidate_allows_none_scores():
    c = Candidate("x", None, None)
    assert c.mean_logprob is None and Candidate.from_dict(c.to_dict()) == c
