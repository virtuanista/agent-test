from app.config import settings
from app.relevance import filter_relevant, parse_scores, reformulate_query
from app.search import Result, is_blocked
from tests.conftest import FakeLLM, run

R = [Result("A", "https://a.com", "x"), Result("B", "https://b.com", "y"), Result("C", "https://c.com", "z")]


def test_parse_scores():
    assert parse_scores("[2, 0, 1]", 3) == [2, 0, 1]
    assert parse_scores("Here: [2,0,1]\nthanks", 3) == [2, 0, 1]
    assert parse_scores("[2, 0]", 3) is None  # wrong length: not trusted
    assert parse_scores("no list", 3) is None
    assert parse_scores("[7, 0, 1]", 3) == [2, 0, 1]  # clamped


def test_filter_keeps_relevant_best_first():
    llm = FakeLLM(relevance="[1,2,0]")
    kept = run(filter_relevant(llm, "goal", "task", R))
    assert [r.title for r in kept] == ["B", "A"]  # score 2 first, 0 dropped


def test_filter_keeps_everything_when_scoring_is_unparsable():
    llm = FakeLLM(relevance="I cannot do that")
    assert run(filter_relevant(llm, "goal", "task", R)) == R


def test_filter_empty_input_makes_no_llm_call():
    llm = FakeLLM()
    assert run(filter_relevant(llm, "g", "t", [])) == []
    assert llm.prompts == []


def test_reformulate_returns_first_line():
    assert run(reformulate_query(FakeLLM(), "g", "t", "bad query")) == "a better query"


def test_blocked_urls():
    assert is_blocked("https://example.org/paper.PDF")
    assert is_blocked("https://www.instagram.com/reel/x")
    assert is_blocked("https://sub.pinterest.com/pin/1")
    assert not is_blocked("https://github.com/shadcn-ui/ui")
    assert not is_blocked("https://notinstagram.com/x")  # suffix match needs a dot boundary
    assert "instagram.com" in settings.block_domains
