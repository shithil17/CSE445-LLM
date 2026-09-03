import json
import re
import statistics
import sys
import time
from pathlib import Path

import requests

from ml_tools import AVAILABLE_TOOLS

LLM_LATENCIES: list[float] = []


def get_latency_stats() -> dict:
    """Summarize timing for local LLM calls."""


    per_call = list(LLM_LATENCIES)
    count = len(per_call)
    total = round(float(sum(per_call)), 4) if per_call else 0.0
    mean = round(float(statistics.mean(per_call)), 4) if per_call else 0.0
    median = round(float(statistics.median(per_call)), 4) if per_call else 0.0
    return {"per_call": per_call, "mean": mean, "median": median, "total": total, "count": count}


# Backwards-compat alias for direct dict-style access
LATENCY_STATS = LLM_LATENCIES

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = """You are an expert Autonomous Machine Learning Assistant.
You solve machine learning problems by thinking step-by-step and invoking external tools.
You have access to the following tools:
1. load_dataset_summary(dataset_name: str) -> JSON summary of dataset (iris, wine, breast_cancer).
2. train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> JSON test/CV scores (model_type: decision_tree, logistic_regression, random_forest).
3. train_pytorch_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01) -> JSON PyTorch training and evaluation results.
4. tune_hyperparameters(dataset_name: str, model_type: str = "svc", cv: int = 5) -> JSON best params + CV score via GridSearchCV (model_type: svc, decision_tree; small grid <=12 candidates).
5. select_features(dataset_name: str, n_features: int = 2, cv: int = 3) -> JSON selected features/names + PCA explained variance (PCA n_components=n_features, forward SequentialFeatureSelector with LogisticRegression, cv 2-10).
6. train_regularized_mlp(dataset_name: str, hidden_dim: int = 64, dropout_rate: float = 0.3, epochs: int = 50, lr: float = 0.01, weight_decay: float = 1e-4, use_batchnorm: bool = True, scheduler: str = "step") -> JSON PyTorch regularized MLP (Dropout+BatchNorm+StepLR/Cosine scheduler, weight_decay) with test_accuracy/final_loss.
7. train_pytorch_mlp_cv(dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01, cv: int = 5) -> JSON with cv_mean_accuracy/cv_std and test_accuracy via real k-fold CV.
To use a tool, you MUST strictly use this format:
Thought: Describe your reasoning about what to do next.
Action: <tool_name>
Action Input: {"param_name": "value"}
After each Action you MUST wait for the system to return Observation: before proposing the next Thought/Action. Do NOT generate Observation: or Received Output: yourself.
When you have received the observation and are ready to provide the complete answer to the user, format your output as:
Thought: I have gathered all necessary experimental data.
Final Answer: <your comprehensive answer with results and recommendation>
Begin!
 """


def query_local_llm(prompt: str) -> str:
    """Send a prompt to the local Ollama server and return its reply.

    Times the request and appends the duration to LLM_LATENCIES for
    reporting. Retries once on transient connection hiccups.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "stop": ["Observation:", "Received Output:"],
        },
    }
    # Retry once on transient EOF / 500 errors (Ollama can hiccup under load)
    for attempt in range(2):
        start = time.time()
        appended = False
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            latency = time.time() - start
            LLM_LATENCIES.append(latency)
            appended = True
            if response.status_code != 200:
                raise RuntimeError(f"Ollama error: {response.text}")
            return response.json().get("response", "")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, RuntimeError) as e:
            if not appended:
                latency = time.time() - start
                LLM_LATENCIES.append(latency)
            if attempt == 0 and ("EOF" in str(e) or "500" in str(e) or "Connection" in str(e) or "Timeout" in str(e)):
                time.sleep(1.5)
                continue
            raise


def _parse_value(raw: str):
    """Turn a raw argument token into a Python value (int/float/bool/None/str)."""


    s = raw.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None
    # int without dot/exponent
    try:
        if "." not in s and "e" not in low and "E" not in s:
            return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s.strip('"\'').strip()


def _split_args_respecting_quotes(s: str) -> list[str]:
    """Split comma-separated args respecting single/double quotes and escapes."""
    parts: list[str] = []
    cur: list[str] = []
    in_single = False
    in_double = False
    escape = False
    for c in s:
        if escape:
            cur.append(c)
            escape = False
            continue
        if c == "\\":
            if in_single or in_double:
                escape = True
            cur.append(c)
            continue
        if c == "'" and not in_double:
            in_single = not in_single
            cur.append(c)
        elif c == '"' and not in_single:
            in_double = not in_double
            cur.append(c)
        elif c == "," and not in_single and not in_double:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _parse_paren_kwargs(inner: str) -> dict:
    """Parse python-style kwargs like 'dataset_name=\"iris\", hidden_dim=32' into dict."""
    inner = inner.strip()
    if not inner:
        return {}
    kwargs: dict = {}
    for part in _split_args_respecting_quotes(inner):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        kwargs[k] = _parse_value(v)
    return kwargs


def _extract_action_paren_kwargs(llm_output: str) -> dict:
    """Extract kwargs from Action line parenthesized args, e.g. Action: tool(a=\"b\", x=1)."""
    # Prefer single-line Action extraction first (most common)
    for line in llm_output.splitlines():
        if "Action:" not in line:
            continue
        m = re.search(r"Action:\s*[a-zA-Z0-9_]+\s*\((.*)\)", line)
        if m:
            inner = m.group(1).strip()
            # Use paren content only if it looks like kwargs (contains =)
            if inner and "=" in inner:
                parsed = _parse_paren_kwargs(inner)
                if parsed:
                    return parsed
    # Fallback: multiline/cross-line Action(...) if LLM wraps args across lines
    m2 = re.search(r"Action:\s*[a-zA-Z0-9_]+\s*\((.*?)\)", llm_output, re.DOTALL)
    if m2:
        inner = m2.group(1).strip()
        if inner and "=" in inner:
            parsed = _parse_paren_kwargs(inner)
            if parsed:
                return parsed
    return {}


def _extract_json_string(llm_output: str) -> str | None:
    """Brace-matching JSON extraction after 'Action Input:' (handles multiline & nested braces)."""
    marker = "Action Input:"
    idx = llm_output.find(marker)
    if idx == -1:
        return None
    start = llm_output.find("{", idx)
    if start == -1:
        return None
    depth = 0
    in_single = False
    in_double = False
    escape = False
    for i in range(start, len(llm_output)):
        c = llm_output[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            if in_single or in_double:
                escape = True
            continue
        if not in_single and not in_double:
            if c == '"':
                in_double = True
            elif c == "'":
                in_single = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return llm_output[start : i + 1]
        else:
            if in_double and c == '"':
                in_double = False
            elif in_single and c == "'":
                in_single = False
    return None


def classify_failure(error_msg: str) -> tuple[str, str]:
    """Sort a tool error into a bucket and suggest a fix."""


    msg = str(error_msg).lower()
    if any(k in msg for k in ["shape", "mismatch", "dimension"]):
        return ("shape mismatch", "Check input dimensions / feature count matches dataset")
    if any(k in msg for k in ["hyperparameter", "invalid", "unsupported", "param", "hidden_dim", "dropout_rate", "cv", "n_features"]):
        return (
            "invalid hyperparameter",
            "Verify parameter ranges: hidden_dim 4-512, dropout 0-0.9, epochs 1-500, cv 2-10, n_features within dataset bounds",
        )
    if any(k in msg for k in ["nan", "inf", "diverged", "finite", "loss"]):
        return ("NaN/inf loss", "Try lower learning rate, lower dropout, or check normalization")
    return ("tool_error", "Check tool arguments and dataset name; ensure parameters are valid")


def run_agent_loop(user_query: str, max_iterations: int = 6):
    print("\n=======================================================")
    print(f"USER QUERY: {user_query}")
    print("=======================================================\n")
    prompt = f"{SYSTEM_PROMPT}\nUser Query: {user_query}\n"
    consecutive_failures = 0
    observation_count = 0
    for step in range(1, max_iterations + 1):
        print(f"\n--- Step {step} ---")
        llm_output = query_local_llm(prompt)
        print(f"[react_agent] LLM response preview: {llm_output[:200].replace(chr(10), ' ')}", file=sys.stderr)
        try:
            Path("logs").mkdir(parents=True, exist_ok=True)
            with open("logs/benchmark_agent_raw.log", "a", encoding="utf-8") as f:
                f.write(f"=== CALL {step} @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(llm_output)
                f.write("\n\n")
        except Exception as e:
            print(f"[benchmark_runner] log write failed: {e}", file=sys.stderr)
        print(llm_output)
        prompt += llm_output
        action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", llm_output)
        raw_json_str = _extract_json_string(llm_output)
        paren_kwargs = _extract_action_paren_kwargs(llm_output)
        # has_tool_call considers either JSON Action Input or parenthesized Action args
        has_tool_call = bool(action_match and (raw_json_str is not None or paren_kwargs))
        # Only treat as Final Answer if no valid Action is present in same output
        # (handles hallucination where model emits both)
        if "Final Answer:" in llm_output and not has_tool_call:
            if observation_count >= 1:
                print("\n>>> Task Completed Successfully!")
                break
            if consecutive_failures >= 2:
                print("\n>>> Task Completed Successfully!")
                break
            consecutive_failures += 1
            observation = "\nObservation: You have not called any tools yet. You must use at least one Action/Action Input tool call before giving a Final Answer. Please select a tool now.\n"
            if consecutive_failures >= 2:
                observation = observation.rstrip("\n") + " Max retries (2) reached for this failure type; please try a different approach or provide a Final Answer after calling at least one tool.\n"
            print(observation)
            prompt += observation
            continue
        if has_tool_call:
            tool_name = action_match.group(1).strip()  # type: ignore[union-attr]
            # Resolve kwargs: prefer JSON, fallback/merge parenthesized args
            raw_input = raw_json_str.strip() if raw_json_str is not None else "{}"
            # If no JSON at all but paren_kwargs present, raw_input is "{}" and we handle below
            kwargs: dict = {}
            json_parse_failed = False
            json_err: Exception | None = None
            if raw_json_str is not None:
                try:
                    parsed = json.loads(raw_input)
                    if isinstance(parsed, dict):
                        kwargs = parsed
                    else:
                        kwargs = {}
                except json.JSONDecodeError as e:
                    json_parse_failed = True
                    json_err = e
                    kwargs = {}
            else:
                # No JSON object found; treat as empty and rely on paren fallback
                json_parse_failed = False
                kwargs = {}

            # Merge / fallback to parenthesized args: if JSON empty/incomplete or parse failed
            if paren_kwargs:
                if json_parse_failed:
                    # Prefer parenthesized args over malformed JSON
                    kwargs = paren_kwargs
                    json_parse_failed = False  # recovered
                elif not kwargs:
                    kwargs = paren_kwargs
                else:
                    # merge missing keys from paren (paren supplements JSON)
                    for k, v in paren_kwargs.items():
                        if k not in kwargs:
                            kwargs[k] = v

            if json_parse_failed:
                err_detail = f"Error parsing Action Input as JSON: {str(json_err)}"
                category, hint = classify_failure(err_detail + " " + raw_input)
                consecutive_failures += 1
                observation = (
                    f"\nObservation: Self-healing: Detected {category}: {err_detail}. "
                    f"Hint: {hint}. Please correct and retry (attempt {consecutive_failures}/2).\n"
                )
                if consecutive_failures >= 2:
                    observation = observation.rstrip("\n") + " Max retries (2) reached for this failure type; please try a different approach or provide a Final Answer.\n"
                prompt += observation
                print(observation)
                continue
            if tool_name in AVAILABLE_TOOLS:
                try:
                    tool_res = AVAILABLE_TOOLS[tool_name](**kwargs)
                    # Detect tool JSON error payload (tools never raise past boundary)
                    payload = None
                    try:
                        payload = json.loads(tool_res)
                    except Exception:
                        payload = None
                    if isinstance(payload, dict) and "error" in payload:
                        err_msg = str(payload.get("error", tool_res))
                        category, hint = classify_failure(err_msg)
                        consecutive_failures += 1
                        observation = (
                            f"\nObservation: Self-healing: Detected {category}: {err_msg}. "
                            f"Hint: {hint}. Please correct and retry (attempt {consecutive_failures}/2).\n"
                        )
                        if consecutive_failures >= 2:
                            observation = observation.rstrip("\n") + " Max retries (2) reached for this failure type; please try a different approach or provide a Final Answer.\n"
                    else:
                        consecutive_failures = 0
                        observation_count += 1
                        observation = f"\nObservation: {tool_res}\n"
                except Exception as e:
                    err_msg = str(e)
                    category, hint = classify_failure(err_msg)
                    consecutive_failures += 1
                    observation = (
                        f"\nObservation: Self-healing: Detected {category}: Tool execution error: {err_msg}. "
                        f"Hint: {hint}. Please correct and retry (attempt {consecutive_failures}/2).\n"
                    )
                    if consecutive_failures >= 2:
                        observation = observation.rstrip("\n") + " Max retries (2) reached for this failure type; please try a different approach or provide a Final Answer.\n"
            else:
                err_msg = f"Tool '{tool_name}' not recognized"
                category, hint = classify_failure(err_msg)
                consecutive_failures += 1
                observation = (
                    f"\nObservation: Self-healing: Detected {category}: {err_msg}. "
                    f"Hint: {hint}. Please correct and retry (attempt {consecutive_failures}/2).\n"
                )
                if consecutive_failures >= 2:
                    observation = observation.rstrip("\n") + " Max retries (2) reached for this failure type; please try a different approach or provide a Final Answer.\n"
            print(observation)
            prompt += observation
        else:
            prompt += "\nObservation: Please respond with an Action and Action Input or a Final Answer.\n"


if __name__ == "__main__":
    test_task = (
        "Analyze the breast_cancer dataset, train a Random Forest and a PyTorch MLP on it, "
        "compare their accuracies, and recommend the best model for clinical screening."
    )
    run_agent_loop(test_task)
