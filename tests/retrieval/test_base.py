from cloudagent.retrieval.base import RetrievalResult


def test_retrieval_result_defaults():
    r = RetrievalResult(content="hello", source="vector")
    assert r.content == "hello"
    assert r.source == "vector"
    assert r.score == 0.0
    assert r.metadata == {}
