from app.parsing import (
    ground_summary,
    is_duplicate,
    parse_new_task,
    parse_scout_plan,
    parse_task_list,
    similarity,
)


def test_parse_task_list_variants():
    assert parse_task_list('["a b", "c"]') == ["a b", "c"]
    assert parse_task_list("1. uno\n2. dos") == ["uno", "dos"]
    assert parse_task_list("Aqui van:\n```json\n['x y']\n```") == ["x y"]


def test_empty_array_is_not_a_task():
    # Regression: "[]" used to fall through to the line parser and become a task called "[]".
    assert parse_task_list("[]") == []


def test_parse_new_task():
    assert parse_new_task("Task: Buscar X") == "Buscar X"
    assert parse_new_task("No new tasks needed") == ""
    assert parse_new_task("") == ""
    assert parse_new_task('"Investigar Y"\nporque...') == "Investigar Y"


# Real tasks from the run that exposed the problem (run 4d53cec9402d).
TASK_2 = "generaciones de diseño web pobres causas y soluciones"
TASK_7 = (
    "investigar causas específicas de las generaciones de diseño web pobres y proponer soluciones "
    "prácticas con enfoque en IA y herramientas de prototipado rápido para mejorar la calidad de UI/UX en 2024."
)


def test_rephrased_task_is_duplicate():
    assert is_duplicate(TASK_7, [TASK_2])
    assert is_duplicate("Shadcn/UI AI workflow", ["shadcn ui ai workflow"])


def test_distinct_tasks_are_not_duplicates():
    assert not is_duplicate("shadcn/ui registry AI generated components", ["taste-skill AI design github"])
    assert not is_duplicate("scroll triggered image sequence animation", [TASK_2])
    assert similarity("", "x") == 0.0


def test_ground_summary_removes_unknown_links():
    text = "Dato[1](https://a.com/x) y otro[[2]](https://evil.com/z) y [[tabla]](##) fin."
    out, stats = ground_summary(text, {"https://a.com/x"})
    assert "[1](https://a.com/x)" in out
    assert "evil.com" not in out and "(##)" not in out and "[[" not in out
    assert "tabla" in out  # text links keep their text, numeric citations vanish
    assert stats["links_kept"] == 1 and stats["links_removed"] == 2


def test_ground_summary_tolerates_trailing_slash_and_case():
    out, stats = ground_summary("x[1](https://A.com/path/)", {"https://a.com/path"})
    assert stats["links_kept"] == 1 and stats["links_removed"] == 0


def test_ground_summary_flags_uncited_claims():
    long = "Esta es una afirmación bastante larga que no lleva ninguna fuente asociada a ella."
    cited = "Esta afirmación sí lleva fuente y es lo bastante larga para contarse[1](https://a.com)."
    out, stats = ground_summary("\n".join([long, long, long, cited]), {"https://a.com"})
    assert stats["uncited_lines"] == 3 and stats["claim_lines"] == 4
    assert "⚠" in out
    out2, _ = ground_summary("\n".join([cited, cited, cited, cited]), {"https://a.com"})
    assert "⚠" not in out2


def test_parse_scout_plan_object():
    text = 'Sure:\n{"problem": "UIs look generic", "web_queries": ["a b", "c d"], "github_queries": ["x y"]}'
    plan = parse_scout_plan(text)
    assert plan == {"problem": "UIs look generic", "web_queries": ["a b", "c d"], "github_queries": ["x y"]}


def test_parse_scout_plan_caps_queries_and_cleans_them():
    text = '{"problem": "p", "web_queries": ["1. one", "two", "three", "four"], "github_queries": []}'
    assert parse_scout_plan(text)["web_queries"] == ["one", "two", "three"]


def test_parse_scout_plan_falls_back_to_a_plain_array():
    plan = parse_scout_plan('["only web", "queries"]')
    assert plan["web_queries"] == ["only web", "queries"] and plan["github_queries"] == [] and plan["problem"] == ""


def test_parse_scout_plan_garbage_is_empty_not_an_error():
    plan = parse_scout_plan("I cannot help with that")
    assert plan["web_queries"] and plan["github_queries"] == []  # line fallback, never raises
    assert parse_scout_plan("") == {"problem": "", "web_queries": [], "github_queries": []}


def test_ground_summary_strips_invisible_characters():
    text = "Dato\u200b[3](https://a.com/x)\u200b y \u2060fin"
    out, stats = ground_summary(text, {"https://a.com/x"})
    assert "\u200b" not in out and "\u2060" not in out
    assert "[3](https://a.com/x)" in out and stats["links_kept"] == 1


def test_bare_numeric_citations_become_links_to_the_numbered_sources():
    srcs = ["https://a.com/x", "https://b.com/y"]
    out, stats = ground_summary("Dato [1] y otro [2]. Ya enlazado [1](https://a.com/x). Fuera de lista [9].", set(srcs), "Spanish", srcs)
    assert "[1](https://a.com/x) y otro [2](https://b.com/y)" in out
    assert out.count("[1](https://a.com/x)") == 2 and "[9]" in out and "[9](" not in out
    assert stats["links_kept"] == 3


def test_bare_citations_are_untouched_without_a_numbered_list():
    out, _ = ground_summary("Dato [1].", {"https://a.com/x"})
    assert "[1]" in out and "](" not in out
