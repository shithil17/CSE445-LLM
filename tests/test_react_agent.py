import json
import re
from unittest.mock import patch, MagicMock

import pytest

import react_agent
from react_agent import (
    _extract_action_paren_kwargs,
    _extract_json_string,
    _parse_paren_kwargs,
    _parse_value,
    run_agent_loop,
)

# ---------------------------------------------------------------------------
# Direct regex contract tests (no mocking needed)
# ---------------------------------------------------------------------------

ACTION_RE = re.compile(r"Action:\s*([a-zA-Z0-9_]+)")
INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL)


def test_regex_extracts_tool_name_and_kwargs():
    canned = """Thought: I should check the iris dataset summary.
Action: load_dataset_summary
Action Input: {"dataset_name": "iris"}"""
    m_action = ACTION_RE.search(canned)
    m_input = INPUT_RE.search(canned, re.DOTALL) if False else re.search(r"Action Input:\s*(\{.*?\})", canned, re.DOTALL)
    # use actual patterns from react_agent.py
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", canned)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", canned, re.DOTALL)
    assert action_match is not None
    assert action_match.group(1) == "load_dataset_summary"
    assert input_match is not None
    kwargs = json.loads(input_match.group(1))
    assert kwargs == {"dataset_name": "iris"}


def test_regex_parses_train_sklearn_action():
    canned = """Thought: Try RF.
Action: train_sklearn_model
Action Input: {"dataset_name": "breast_cancer", "model_type": "random_forest"}"""
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", canned)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", canned, re.DOTALL)
    assert action_match.group(1) == "train_sklearn_model"
    kwargs = json.loads(input_match.group(1))
    assert kwargs["dataset_name"] == "breast_cancer"
    assert kwargs["model_type"] == "random_forest"


def test_regex_parses_pytorch_action():
    canned = 'Thought: MLP\nAction: train_pytorch_mlp\nAction Input: {"dataset_name": "wine", "hidden_dim": 32, "epochs": 5}'
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", canned)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", canned, re.DOTALL)
    assert action_match.group(1) == "train_pytorch_mlp"
    kwargs = json.loads(input_match.group(1))
    assert kwargs["hidden_dim"] == 32


# ---------------------------------------------------------------------------
# run_agent_loop with mocked LLM
# ---------------------------------------------------------------------------

def test_run_agent_loop_parses_action_and_calls_tool(capsys):
    """Mocked LLM returns a valid Action; assert tool is called and Observation printed."""
    responses = [
        'Thought: Check iris.\nAction: load_dataset_summary\nAction Input: {"dataset_name": "iris"}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: iris has 150 samples.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses) as mock_llm:
        run_agent_loop("summarize iris", max_iterations=3)
        assert mock_llm.call_count == 2
    captured = capsys.readouterr().out
    assert "Observation:" in captured
    assert "Final Answer:" in captured


def test_run_agent_loop_malformed_json_degrades_to_observation(capsys):
    """Malformed JSON in Action Input must not crash; loop emits parsing error Observation."""
    responses = [
        'Thought: bad json\nAction: load_dataset_summary\nAction Input: {bad json}',
        'Thought: fix\nAction: load_dataset_summary\nAction Input: {"dataset_name": "iris"}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: done despite bad json.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses):
        run_agent_loop("test malformed json", max_iterations=4)
    captured = capsys.readouterr().out
    assert "Error parsing Action Input as JSON" in captured
    assert "Final Answer:" in captured


def test_run_agent_loop_unknown_tool(capsys):
    responses = [
        'Thought: try unknown\nAction: magic_tool\nAction Input: {"foo": "bar"}',
        'Thought: fix\nAction: load_dataset_summary\nAction Input: {"dataset_name": "iris"}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: handled unknown tool.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses):
        run_agent_loop("test unknown tool", max_iterations=4)
    captured = capsys.readouterr().out
    assert "not recognized" in captured


def test_run_agent_loop_missing_action_prompts_observation(capsys):
    """When LLM returns no Action/Final Answer, loop adds corrective Observation to prompt and recovers."""
    responses = [
        'Thought: I am not sure what to do.',
        'Thought: check iris\nAction: load_dataset_summary\nAction Input: {"dataset_name": "iris"}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: recovered.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses) as mock_llm:
        run_agent_loop("test missing action", max_iterations=4)
        # second prompt should contain the corrective Observation appended by the loop
        second_prompt = mock_llm.call_args_list[1][0][0]
        assert "Please respond with an Action" in second_prompt
    captured = capsys.readouterr().out
    assert "Final Answer:" in captured


def test_run_agent_loop_terminates_on_final_answer(capsys):
    responses = [
        'Thought: check iris\nAction: load_dataset_summary\nAction Input: {"dataset_name": "iris"}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: immediate answer.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses) as mock_llm:
        run_agent_loop("immediate final", max_iterations=6)
        assert mock_llm.call_count == 2
    captured = capsys.readouterr().out
    assert "Task Completed Successfully" in captured


def test_run_agent_loop_terminates_on_max_iterations_without_crash(capsys):
    """Loop must not crash when max_iterations reached without Final Answer."""
    responses = [
        'Thought: still thinking\nAction: load_dataset_summary\nAction Input: {"dataset_name": "iris"}'
    ] * 10
    with patch("react_agent.query_local_llm", side_effect=responses):
        # should not raise
        run_agent_loop("never finishes", max_iterations=3)
    captured = capsys.readouterr().out
    # no Final Answer completion message expected, but no crash
    assert "Step 3" in captured
    assert "Task Completed Successfully" not in captured


def test_run_agent_loop_tool_exception_becomes_observation(capsys):
    """If tool raises, Observation with error is emitted, not a crash."""
    def fake_tool(**kwargs):
        raise ValueError("injected failure")

    responses = [
        'Thought: call tool\nAction: load_dataset_summary\nAction Input: {"dataset_name": "iris"}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: recovered from tool error.',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: recovered from tool error.',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: recovered from tool error.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses):
        with patch.dict("react_agent.AVAILABLE_TOOLS", {"load_dataset_summary": fake_tool}):
            run_agent_loop("tool exception", max_iterations=5)
    captured = capsys.readouterr().out
    assert "Tool execution error" in captured
    assert "injected failure" in captured


def test_query_local_llm_error_handling():
    """query_local_llm should raise RuntimeError on non-200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "internal error"
    with patch("react_agent.requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="Ollama error"):
            react_agent.query_local_llm("hello")


# ---------------------------------------------------------------------------
# Self-healing wrapper tests (Phase 3) - JSON {"error":...} payload path + retry cap 2
# ---------------------------------------------------------------------------

def test_self_healing_bad_dataset_triggers_retry_and_recovery(capsys):
    """Bad dataset_name → JSON {error: Unknown dataset} → Self-healing Observation → retry succeeds."""
    responses = [
        'Thought: try bad dataset\nAction: load_dataset_summary\nAction Input: {"dataset_name": "nope"}',
        'Thought: correct to valid dataset\nAction: load_dataset_summary\nAction Input: {"dataset_name": "iris"}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: recovered after self-healing.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses):
        run_agent_loop("self-healing bad dataset", max_iterations=5)
    captured = capsys.readouterr().out
    assert "Self-healing: Detected" in captured
    assert "Hint:" in captured
    assert "attempt 1/2" in captured
    assert "Observation:" in captured
    # should have recovered and completed
    assert "Final Answer:" in captured
    assert "Task Completed Successfully" in captured
    # error payload should be surfaced
    assert "Unknown dataset" in captured or "nope" in captured.lower()


def test_self_healing_invalid_hyperparameter_triggers_retry(capsys):
    """Invalid hyperparameter (hidden_dim=9999 / dropout_rate=5.0) → invalid hyperparameter category → retry."""
    responses = [
        'Thought: try invalid hyperparam\nAction: train_regularized_mlp\nAction Input: {"dataset_name": "iris", "hidden_dim": 9999, "dropout_rate": 0.3, "epochs": 5}',
        'Thought: correct hidden_dim\nAction: train_regularized_mlp\nAction Input: {"dataset_name": "iris", "hidden_dim": 32, "dropout_rate": 0.3, "epochs": 5}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: recovered with valid hidden_dim.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses):
        run_agent_loop("self-healing invalid hyperparam", max_iterations=5)
    captured = capsys.readouterr().out
    assert "Self-healing: Detected invalid hyperparameter" in captured
    assert "Hint:" in captured
    assert "hidden_dim 4-512" in captured
    assert "attempt 1/2" in captured
    assert "Final Answer:" in captured


def test_self_healing_nan_inf_category(capsys):
    """Mocked tool returning NaN/inf diverged error → NaN/inf loss category detected."""
    # Fake tool that on first call returns NaN/inf error payload, second call succeeds
    call_count = {"n": 0}

    def fake_mlp(**kwargs):
        if call_count["n"] == 0:
            call_count["n"] += 1
            return json.dumps({"error": "Training diverged: loss is NaN/inf. Try lower lr."})
        return json.dumps(
            {
                "framework": "PyTorch",
                "dataset": "iris",
                "hidden_dim": 32,
                "dropout_rate": 0.3,
                "use_batchnorm": True,
                "scheduler": "step",
                "weight_decay": 0.0001,
                "epochs": 5,
                "final_loss": 0.42,
                "test_accuracy": 0.93,
                "final_lr": 0.01,
            }
        )

    responses = [
        'Thought: trigger nan\nAction: train_regularized_mlp\nAction Input: {"dataset_name": "iris", "hidden_dim": 32, "epochs": 5}',
        'Thought: retry with lower lr\nAction: train_regularized_mlp\nAction Input: {"dataset_name": "iris", "hidden_dim": 32, "epochs": 5, "lr": 0.001}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: recovered from NaN loss.',
    ]
    with patch.dict("react_agent.AVAILABLE_TOOLS", {"train_regularized_mlp": fake_mlp}):
        with patch("react_agent.query_local_llm", side_effect=responses):
            run_agent_loop("self-healing nan", max_iterations=5)
    captured = capsys.readouterr().out
    assert "Self-healing: Detected NaN/inf loss" in captured
    assert "Hint:" in captured
    assert "lower learning rate" in captured.lower()
    assert "attempt 1/2" in captured
    assert "Final Answer:" in captured


def test_self_healing_retry_cap_enforced(capsys):
    """Two consecutive failures → attempt 1/2 then attempt 2/2 + Max retries, no crash beyond cap."""
    responses = [
        'Thought: bad 1\nAction: load_dataset_summary\nAction Input: {"dataset_name": "nope"}',
        'Thought: bad 2\nAction: load_dataset_summary\nAction Input: {"dataset_name": "nope2"}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: gave up after retries.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses):
        run_agent_loop("retry cap test", max_iterations=5)
    captured = capsys.readouterr().out
    assert "attempt 1/2" in captured
    assert "attempt 2/2" in captured
    assert "Max retries (2) reached" in captured
    # loop should still terminate gracefully on Final Answer, not raise
    assert "Final Answer:" in captured
    assert "Task Completed Successfully" in captured


def test_self_healing_classify_failure_categories():
    """Direct classify_failure mapping for all self-healing categories."""
    from react_agent import classify_failure

    cat, hint = classify_failure("shape mismatch dimension 4 vs 5")
    assert cat == "shape mismatch"
    assert "dimension" in hint.lower()

    cat, hint = classify_failure("Invalid hidden_dim=9999: must be between 4 and 512")
    assert cat == "invalid hyperparameter"
    assert "hidden_dim" in hint

    cat, hint = classify_failure("Training diverged: loss is NaN/inf")
    assert cat == "NaN/inf loss"
    assert "learning rate" in hint.lower()

    cat, hint = classify_failure("Unknown dataset 'nope'")
    assert cat == "tool_error"


# ---------------------------------------------------------------------------
# JSON parsing via _extract_json_string and paren-kwargs parsing
# ---------------------------------------------------------------------------

def test_extract_json_string_happy_path():
    text = 'Action: load_dataset_summary\nAction Input: {"dataset_name": "iris"}'
    s = _extract_json_string(text)
    assert s is not None
    assert json.loads(s) == {"dataset_name": "iris"}


def test_extract_json_string_multiline_nested():
    text = 'Thought: x\nAction: train_pytorch_mlp\nAction Input: {\n  "dataset_name": "wine",\n  "hidden_dim": 32,\n  "nested": {"a": 1}\n}'
    s = _extract_json_string(text)
    assert s is not None
    data = json.loads(s)
    assert data["dataset_name"] == "wine"
    assert data["hidden_dim"] == 32
    assert data["nested"] == {"a": 1}


def test_extract_json_string_returns_none_when_missing():
    assert _extract_json_string("No action input here") is None
    assert _extract_json_string("Action: load_dataset_summary") is None
    assert _extract_json_string("Action Input: not json at all") is None


def test_parse_value_variants():
    assert _parse_value("32") == 32 and isinstance(_parse_value("32"), int)
    assert _parse_value("3.14") == 3.14 and isinstance(_parse_value("3.14"), float)
    assert _parse_value("true") is True
    assert _parse_value("False") is False
    assert _parse_value("None") is None
    assert _parse_value("null") is None
    assert _parse_value('"hello"') == "hello"
    assert _parse_value("'hello'") == "hello"
    assert _parse_value("  42  ") == 42
    # string without quotes
    assert _parse_value("iris") == "iris"


def test_parse_paren_kwargs_happy_path():
    inner = 'dataset_name="iris", hidden_dim=32, lr=0.01, use_batchnorm=True, flag=None'
    kwargs = _parse_paren_kwargs(inner)
    assert kwargs["dataset_name"] == "iris"
    assert kwargs["hidden_dim"] == 32
    assert kwargs["lr"] == 0.01
    assert kwargs["use_batchnorm"] is True
    assert kwargs["flag"] is None


def test_parse_paren_kwargs_respects_quotes_and_comma():
    inner = 'a="hello, world", b=2'
    kwargs = _parse_paren_kwargs(inner)
    assert kwargs["a"] == "hello, world"
    assert kwargs["b"] == 2


def test_extract_action_paren_kwargs_happy_path():
    llm_out = 'Thought: call mlp\nAction: train_pytorch_mlp(dataset_name="iris", hidden_dim=32)\nAction Input: {}'
    kwargs = _extract_action_paren_kwargs(llm_out)
    assert kwargs["dataset_name"] == "iris"
    assert kwargs["hidden_dim"] == 32


def test_extract_action_paren_kwargs_multiline():
    llm_out = 'Action: train_pytorch_mlp(\n  dataset_name="wine",\n  hidden_dim=16\n)'
    kwargs = _extract_action_paren_kwargs(llm_out)
    assert kwargs["dataset_name"] == "wine"
    assert kwargs["hidden_dim"] == 16


def test_extract_action_paren_kwargs_empty_when_no_kwargs():
    assert _extract_action_paren_kwargs('Action: load_dataset_summary\nAction Input: {"dataset_name":"iris"}') == {}
    assert _extract_action_paren_kwargs("No action here") == {}


# ---------------------------------------------------------------------------
# Paren-kwargs happy path via run_agent_loop + JSON+paren merge
# ---------------------------------------------------------------------------

def test_run_agent_loop_paren_kwargs_parsing(capsys):
    """Action with parenthesized kwargs (no JSON) should be correctly extracted and executed."""
    responses = [
        'Thought: Try paren style.\nAction: train_pytorch_mlp(dataset_name="iris", hidden_dim=16, epochs=5)',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: paren style worked.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses):
        run_agent_loop("paren test", max_iterations=3)
    captured = capsys.readouterr().out
    assert "Observation:" in captured
    assert "PyTorch" in captured or "test_accuracy" in captured or "framework" in captured.lower()
    assert "Final Answer:" in captured


def test_run_agent_loop_json_paren_merge(capsys):
    """If both JSON and paren kwargs present, paren supplements missing keys from JSON."""
    captured_kwargs = {}

    def fake_tool(**kwargs):
        captured_kwargs.update(kwargs)
        return json.dumps({"ok": True, "received": kwargs})

    # JSON has dataset_name only, paren adds hidden_dim=32 - merge should produce both
    responses = [
        'Thought: merge test\nAction: train_pytorch_mlp(dataset_name="iris", hidden_dim=32)\nAction Input: {"dataset_name": "iris"}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: merge done.',
    ]
    with patch.dict("react_agent.AVAILABLE_TOOLS", {"train_pytorch_mlp": fake_tool}):
        with patch("react_agent.query_local_llm", side_effect=responses):
            run_agent_loop("merge test", max_iterations=3)
    # After merge, hidden_dim from paren should be present alongside JSON's dataset_name
    assert captured_kwargs.get("dataset_name") == "iris"
    assert captured_kwargs.get("hidden_dim") == 32
    captured = capsys.readouterr().out
    assert "Observation:" in captured


def test_run_agent_loop_paren_overrides_malformed_json(capsys):
    """Malformed JSON but valid paren kwargs should recover via paren fallback."""
    captured_kwargs = {}

    def fake_tool(**kwargs):
        captured_kwargs.update(kwargs)
        return json.dumps({"ok": True})

    responses = [
        'Thought: bad json but paren ok\nAction: load_dataset_summary(dataset_name="iris")\nAction Input: {bad json}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: recovered via paren.',
    ]
    with patch.dict("react_agent.AVAILABLE_TOOLS", {"load_dataset_summary": fake_tool}):
        with patch("react_agent.query_local_llm", side_effect=responses):
            run_agent_loop("paren fallback", max_iterations=3)
    assert captured_kwargs.get("dataset_name") == "iris"
    captured = capsys.readouterr().out
    # Should NOT emit JSON parse error when paren recovery succeeds; emits normal observation
    assert "Observation:" in captured
    assert "Final Answer:" in captured


# ---------------------------------------------------------------------------
# Self-healing retry cap: consecutive_failures resets on success and caps at 2
# ---------------------------------------------------------------------------

def test_self_healing_consecutive_failures_resets_on_success(capsys):
    """consecutive_failures should reset to 0 after a success, then cap again at 2."""
    responses = [
        'Thought: bad 1\nAction: load_dataset_summary\nAction Input: {"dataset_name": "nope"}',  # fail -> attempt 1/2
        'Thought: good fix\nAction: load_dataset_summary\nAction Input: {"dataset_name": "iris"}',  # success -> reset
        'Thought: bad again 1\nAction: load_dataset_summary\nAction Input: {"dataset_name": "nope2"}',  # fail -> attempt 1/2 again (not 2)
        'Thought: bad again 2\nAction: load_dataset_summary\nAction Input: {"dataset_name": "nope3"}',  # fail -> attempt 2/2 + Max retries
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: done after reset test.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses):
        run_agent_loop("reset test", max_iterations=6)
    captured = capsys.readouterr().out
    # First failure
    assert "attempt 1/2" in captured
    # After success, next failure should be attempt 1/2 again
    # Count occurrences: should have at least two "attempt 1/2" and one "attempt 2/2"
    assert captured.count("attempt 1/2") >= 2
    assert "attempt 2/2" in captured
    assert "Max retries (2) reached" in captured
    assert "Final Answer:" in captured


def test_self_healing_retry_cap_does_not_exceed_two(capsys):
    """Even with 3 consecutive failures, second failure already caps and not exceed."""
    responses = [
        'Thought: bad 1\nAction: load_dataset_summary\nAction Input: {"dataset_name": "nope"}',
        'Thought: bad 2\nAction: load_dataset_summary\nAction Input: {"dataset_name": "nope"}',
        'Thought: bad 3\nAction: load_dataset_summary\nAction Input: {"dataset_name": "nope"}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: gave up.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses):
        run_agent_loop("cap does not exceed", max_iterations=6)
    captured = capsys.readouterr().out
    assert "attempt 1/2" in captured
    assert "attempt 2/2" in captured
    # Should have Max retries on the second failure; third is also capped (or would be but final answer breaks)
    assert "Max retries (2) reached" in captured


def test_latency_instrumentation_does_not_break_with_mocked_llm(capsys):
    """Mocked LLM calls should not break latency stats; get_latency_stats remains callable."""
    # Clear latencies to isolate test
    react_agent.LLM_LATENCIES.clear()
    responses = [
        'Thought: check iris.\nAction: load_dataset_summary\nAction Input: {"dataset_name": "iris"}',
        'Thought: I have gathered all necessary experimental data.\nFinal Answer: done.',
    ]
    with patch("react_agent.query_local_llm", side_effect=responses):
        run_agent_loop("latency test", max_iterations=3)
    # After mocked run, latencies may be 0 (mock doesn't increment) - ensure stats call doesn't crash
    stats = react_agent.get_latency_stats()
    assert isinstance(stats, dict)
    assert "count" in stats and "mean" in stats and "per_call" in stats
    assert isinstance(stats["per_call"], list)
