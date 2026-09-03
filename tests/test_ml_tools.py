import json

import pytest

from ml_tools import (
    AVAILABLE_TOOLS,
    RegularizedMLP,
    load_dataset_summary,
    select_features,
    train_pytorch_mlp,
    train_pytorch_mlp_cv,
    train_regularized_mlp,
    train_sklearn_model,
    tune_hyperparameters,
)

DATASETS = ["iris", "wine", "breast_cancer"]

# ---------------------------------------------------------------------------
# load_dataset_summary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
def test_load_dataset_summary_valid_json_and_keys(dataset):
    raw = load_dataset_summary(dataset)
    data = json.loads(raw)
    assert "error" not in data, f"unexpected error for {dataset}: {data}"
    for key in ("dataset", "n_samples", "n_features", "feature_names", "classes", "missing_values"):
        assert key in data, f"missing key '{key}' for {dataset}"
    assert data["dataset"] == dataset
    assert isinstance(data["n_samples"], int) and data["n_samples"] > 0
    assert isinstance(data["n_features"], int) and data["n_features"] > 0
    assert isinstance(data["feature_names"], list) and len(data["feature_names"]) == data["n_features"]
    assert isinstance(data["classes"], list) and len(data["classes"]) >= 2
    assert data["missing_values"] == 0


def test_load_dataset_summary_known_sizes():
    assert json.loads(load_dataset_summary("iris"))["n_samples"] == 150
    assert json.loads(load_dataset_summary("wine"))["n_samples"] == 178
    assert json.loads(load_dataset_summary("breast_cancer"))["n_samples"] == 569


def test_load_dataset_summary_unknown_dataset():
    raw = load_dataset_summary("unknown_xyz")
    data = json.loads(raw)
    assert "error" in data
    assert "Unknown dataset" in data["error"]


# ---------------------------------------------------------------------------
# train_sklearn_model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("model_type", ["decision_tree", "logistic_regression", "random_forest"])
def test_train_sklearn_model_valid_json_and_keys(dataset, model_type):
    raw = train_sklearn_model(dataset, model_type)
    data = json.loads(raw)
    assert "error" not in data, f"unexpected error: {data}"
    for key in ("model", "dataset", "test_accuracy", "cv_mean_accuracy", "cv_std"):
        assert key in data
    assert data["model"] == model_type
    assert data["dataset"] == dataset
    assert 0.0 <= data["test_accuracy"] <= 1.0
    assert 0.0 <= data["cv_mean_accuracy"] <= 1.0
    assert 0.0 <= data["cv_std"] <= 0.5


def test_train_sklearn_model_unsupported_model():
    data = json.loads(train_sklearn_model("iris", "unsupported_model"))
    assert "error" in data
    assert "Unsupported model" in data["error"]


def test_train_sklearn_model_unknown_dataset():
    data = json.loads(train_sklearn_model("nope", "random_forest"))
    assert "error" in data


# ---------------------------------------------------------------------------
# train_pytorch_mlp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
def test_train_pytorch_mlp_valid_json_and_keys(dataset):
    # use small epochs/hidden_dim for speed; still exercises full code path
    raw = train_pytorch_mlp(dataset, hidden_dim=16, epochs=5, lr=0.01)
    data = json.loads(raw)
    assert "error" not in data, f"unexpected error: {data}"
    for key in ("framework", "dataset", "hidden_dim", "epochs", "final_loss", "test_accuracy"):
        assert key in data
    assert data["framework"] == "PyTorch"
    assert data["dataset"] == dataset
    assert data["hidden_dim"] == 16
    assert data["epochs"] == 5
    assert isinstance(data["final_loss"], float)
    assert 0.0 <= data["test_accuracy"] <= 1.0
    # loss should be finite and reasonable
    assert data["final_loss"] >= 0.0
    assert data["final_loss"] < 10.0


def test_train_pytorch_mlp_unknown_dataset():
    data = json.loads(train_pytorch_mlp("nope"))
    assert "error" in data


def test_train_pytorch_mlp_deterministic_keys():
    raw = train_pytorch_mlp("iris", hidden_dim=8, epochs=2)
    data = json.loads(raw)
    assert data["hidden_dim"] == 8
    assert data["epochs"] == 2


# ---------------------------------------------------------------------------
# train_pytorch_mlp_cv (Phase 2 new tool: real k-fold CV)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
def test_train_pytorch_mlp_cv_valid_json_and_keys(dataset):
    raw = train_pytorch_mlp_cv(dataset, hidden_dim=16, epochs=5, lr=0.01, cv=3)
    data = json.loads(raw)
    assert "error" not in data, f"unexpected error for {dataset}: {data}"
    for key in ("framework", "dataset", "hidden_dim", "epochs", "lr", "cv", "cv_mean_accuracy", "cv_std", "test_accuracy", "final_loss"):
        assert key in data, f"missing key '{key}' for {dataset}"
    assert data["framework"] == "PyTorch"
    assert data["dataset"] == dataset
    assert data["hidden_dim"] == 16
    assert data["epochs"] == 5
    assert data["cv"] == 3
    assert 0.0 <= data["cv_mean_accuracy"] <= 1.0
    assert 0.0 <= data["cv_std"] <= 0.5
    assert 0.0 <= data["test_accuracy"] <= 1.0
    assert isinstance(data["final_loss"], float)
    assert data["final_loss"] >= 0.0
    assert data["final_loss"] < 10.0


def test_train_pytorch_mlp_cv_unknown_dataset():
    data = json.loads(train_pytorch_mlp_cv("nope", hidden_dim=16, epochs=5))
    assert "error" in data
    assert "not found" in data["error"].lower() or "unknown" in data["error"].lower()


def test_train_pytorch_mlp_cv_invalid_hidden_dim():
    # hidden_dim=1 below valid range 4-512 should return {"error": ...}
    data = json.loads(train_pytorch_mlp_cv("iris", hidden_dim=1, epochs=5))
    assert "error" in data
    assert "hidden_dim" in data["error"].lower() or "invalid" in data["error"].lower()


def test_train_pytorch_mlp_cv_invalid_cv():
    # cv=99 outside 2-10 should return {"error": ...}
    data = json.loads(train_pytorch_mlp_cv("iris", hidden_dim=16, epochs=5, cv=99))
    assert "error" in data
    assert "cv" in data["error"].lower() or "invalid" in data["error"].lower()


def test_train_pytorch_mlp_cv_invalid_lr_and_epochs():
    # invalid lr >1 and epochs out of range should also surface error
    data = json.loads(train_pytorch_mlp_cv("iris", hidden_dim=16, epochs=5, lr=5.0))
    assert "error" in data
    data2 = json.loads(train_pytorch_mlp_cv("iris", hidden_dim=16, epochs=0))
    assert "error" in data2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_available_tools_registry():
    # Phase 2+ : registry must contain all 7 tools (including pytorch_mlp_cv)
    expected = {
        "load_dataset_summary",
        "train_sklearn_model",
        "train_pytorch_mlp",
        "tune_hyperparameters",
        "select_features",
        "train_regularized_mlp",
        "train_pytorch_mlp_cv",
    }
    assert expected == set(AVAILABLE_TOOLS.keys()), f"registry drift: {set(AVAILABLE_TOOLS.keys())}"
    for name, fn in AVAILABLE_TOOLS.items():
        assert callable(fn), f"{name} not callable"


def test_system_prompt_matches_registry():
    """SYSTEM_PROMPT tool list must stay in sync with AVAILABLE_TOOLS (no drift)."""
    from react_agent import SYSTEM_PROMPT

    for name in AVAILABLE_TOOLS:
        assert name in SYSTEM_PROMPT, f"tool '{name}' missing from SYSTEM_PROMPT"
    # sanity: no phantom tool listed that isn't in registry (check count hints)
    # We expect at least 6 tool mentions; not strict exact parse.


# ---------------------------------------------------------------------------
# tune_hyperparameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("model_type", ["svc", "decision_tree"])
def test_tune_hyperparameters_valid_json_and_keys(dataset, model_type):
    raw = tune_hyperparameters(dataset, model_type, cv=3)
    data = json.loads(raw)
    assert "error" not in data, f"unexpected error: {data}"
    for key in ("model", "dataset", "best_params", "best_cv_score", "cv", "n_candidates"):
        assert key in data, f"missing key '{key}'"
    assert data["dataset"] == dataset
    assert 0.0 <= data["best_cv_score"] <= 1.0
    assert data["cv"] == 3
    assert isinstance(data["best_params"], dict) and len(data["best_params"]) > 0
    assert data["n_candidates"] in (6, 12)  # svc=6, dt=12


def test_tune_hyperparameters_aliases():
    # svm alias and case-insensitivity
    data = json.loads(tune_hyperparameters("iris", "svm", cv=3))
    assert "error" not in data
    assert data["model"] == "svc"
    data = json.loads(tune_hyperparameters("iris", "SVM", cv=3))
    assert "error" not in data


def test_tune_hyperparameters_unknown_dataset():
    data = json.loads(tune_hyperparameters("nope", "svc"))
    assert "error" in data


def test_tune_hyperparameters_invalid_model():
    data = json.loads(tune_hyperparameters("iris", "not_a_model"))
    assert "error" in data


def test_tune_hyperparameters_invalid_cv():
    data = json.loads(tune_hyperparameters("iris", "svc", cv=99))
    assert "error" in data


# ---------------------------------------------------------------------------
# select_features
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
def test_select_features_valid_json_and_keys(dataset):
    raw = select_features(dataset, n_features=2, cv=3)
    data = json.loads(raw)
    assert "error" not in data, f"unexpected error: {data}"
    for key in (
        "dataset",
        "n_samples",
        "n_features_original",
        "n_selected",
        "cv",
        "selected_features",
        "selected_indices",
        "pca_explained_variance_ratio",
        "pca_cumulative_variance",
        "selector_cv_mean_accuracy",
        "selector_cv_std",
    ):
        assert key in data, f"missing key '{key}'"
    assert data["dataset"] == dataset
    assert data["n_selected"] == 2
    assert len(data["selected_features"]) == 2
    assert len(data["selected_indices"]) == 2
    assert len(data["pca_explained_variance_ratio"]) == 2
    assert 0.0 <= data["pca_cumulative_variance"] <= 1.0
    assert 0.0 <= data["selector_cv_mean_accuracy"] <= 1.0


def test_select_features_n_features_one():
    data = json.loads(select_features("iris", n_features=1, cv=3))
    assert "error" not in data
    assert data["n_selected"] == 1
    assert len(data["selected_features"]) == 1


def test_select_features_unknown_dataset():
    data = json.loads(select_features("nope", n_features=2))
    assert "error" in data


def test_select_features_invalid_n_features():
    data = json.loads(select_features("iris", n_features=99))
    assert "error" in data


def test_select_features_invalid_cv():
    data = json.loads(select_features("iris", n_features=2, cv=99))
    assert "error" in data


# ---------------------------------------------------------------------------
# train_regularized_mlp (Phase 2 deep classifier with Dropout/BatchNorm/Scheduler)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
def test_train_regularized_mlp_valid_json_and_keys(dataset):
    raw = train_regularized_mlp(dataset, hidden_dim=16, dropout_rate=0.2, epochs=5, lr=0.01)
    data = json.loads(raw)
    assert "error" not in data, f"unexpected error: {data}"
    for key in (
        "framework",
        "dataset",
        "hidden_dim",
        "dropout_rate",
        "use_batchnorm",
        "scheduler",
        "weight_decay",
        "epochs",
        "final_loss",
        "test_accuracy",
        "final_lr",
    ):
        assert key in data, f"missing key '{key}'"
    assert data["framework"] == "PyTorch"
    assert data["dataset"] == dataset
    assert data["hidden_dim"] == 16
    assert data["epochs"] == 5
    assert 0.0 <= data["test_accuracy"] <= 1.0
    assert data["final_loss"] >= 0.0
    assert data["final_loss"] < 10.0


def test_train_regularized_mlp_scheduler_variants():
    # step scheduler (default)
    data_step = json.loads(train_regularized_mlp("iris", hidden_dim=16, epochs=5, scheduler="step"))
    assert data_step["scheduler"] == "step"
    # cosine variant
    data_cos = json.loads(train_regularized_mlp("iris", hidden_dim=16, epochs=5, scheduler="cosine"))
    assert data_cos["scheduler"] == "cosine"
    assert "error" not in data_cos


def test_train_regularized_mlp_no_batchnorm_no_dropout():
    data = json.loads(train_regularized_mlp("iris", hidden_dim=16, dropout_rate=0.0, use_batchnorm=False, epochs=5))
    assert "error" not in data
    assert data["use_batchnorm"] is False
    assert data["dropout_rate"] == 0.0


def test_train_regularized_mlp_weight_decay_and_dropout_params():
    # verify weight_decay passthrough and dropout param echoed
    data = json.loads(train_regularized_mlp("wine", hidden_dim=32, dropout_rate=0.5, weight_decay=0.01, epochs=5))
    assert "error" not in data
    assert data["dropout_rate"] == 0.5
    assert data["weight_decay"] == 0.01


def test_train_regularized_mlp_unknown_dataset():
    data = json.loads(train_regularized_mlp("nope"))
    assert "error" in data


def test_train_regularized_mlp_invalid_dropout():
    data = json.loads(train_regularized_mlp("iris", dropout_rate=5.0))
    assert "error" in data


def test_train_regularized_mlp_invalid_scheduler():
    data = json.loads(train_regularized_mlp("iris", scheduler="notascheduler"))
    assert "error" in data


def test_train_regularized_mlp_invalid_hidden_dim():
    data = json.loads(train_regularized_mlp("iris", hidden_dim=1))
    assert "error" in data


def test_regularized_mlp_module_has_dropout_batchnorm_and_scheduler():
    """Structural: module must contain Dropout + BatchNorm1d, and file must reference schedulers."""
    import inspect
    import ml_tools as m

    src = inspect.getsource(m.RegularizedMLP)
    assert "Dropout" in src
    assert "BatchNorm" in src
    src_tool = inspect.getsource(m.train_regularized_mlp)
    assert "StepLR" in src_tool or "StepLR" in open(m.__file__).read()
    assert "CosineAnnealingLR" in open(m.__file__).read() or "Cosine" in src_tool
    # instantiate and check layers
    net = RegularizedMLP(input_dim=4, hidden_dim=8, num_classes=3, dropout_rate=0.3, use_batchnorm=True)
    has_bn = any(isinstance(l, m.torch.nn.BatchNorm1d) for l in net.net)
    has_do = any(isinstance(l, m.torch.nn.Dropout) for l in net.net)
    assert has_bn, "RegularizedMLP missing BatchNorm1d"
    assert has_do, "RegularizedMLP missing Dropout"
