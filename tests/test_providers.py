from providers import DryRunProvider, ModelResponse, build_provider, load_roles
from providers.base import Usage


def test_dry_run_handler_mode():
    p = DryRunProvider(handler=lambda msgs: ModelResponse(text="hi"))
    r = p.complete([{"role": "user", "content": "yo"}])
    assert r.text == "hi"
    assert r.usage.total_tokens > 0  # nominal cost stamped


def test_dry_run_responses_cycle():
    p = DryRunProvider(responses=[ModelResponse(text="a"), ModelResponse(text="b")])
    assert [p.complete([]).text for _ in range(3)] == ["a", "b", "a"]


def test_build_provider_dry_run():
    assert build_provider({"provider": "dry_run"}).name == "dry_run"


def test_load_roles_constructs_without_network(tmp_path):
    yaml_text = """
roles:
  online:
    provider: dry_run
  offline:
    provider: openai_compat
    base_url: http://localhost:11434/v1
    model: qwen2.5:14b-instruct
"""
    f = tmp_path / "models.yaml"
    f.write_text(yaml_text)
    roles = load_roles(f)
    assert set(roles) == {"online", "offline"}
    assert "qwen2.5" in roles["offline"].name
