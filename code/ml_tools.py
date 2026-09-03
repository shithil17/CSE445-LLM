import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def load_dataset_summary(dataset_name: str) -> str:
    """Return a quick JSON summary for a bundled sklearn dataset.

    Supports iris, wine and breast_cancer. Reports sample count, feature
    names, class labels and missing-value count.
    """
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Unknown dataset '{name}'. Options: {list(DATASETS.keys())}"})
    data = DATASETS[name]()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    summary = {
        "dataset": name,
        "n_samples": df.shape[0],
        "n_features": len(data.feature_names),
        "feature_names": list(data.feature_names),
        "classes": [str(c) for c in np.unique(data.target)],
        "missing_values": int(df.isnull().sum().sum()),
    }
    return json.dumps(summary)


def train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> str:
    """Fit a sklearn classifier and report accuracy.

    Handles decision_tree, logistic_regression and random_forest. Splits
    the dataset, trains, then returns test accuracy plus 5-fold CV mean/std.
    """
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})
    data = DATASETS[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=42, stratify=data.target
    )
    model_type = model_type.lower().strip()
    if model_type == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        return json.dumps({"error": f"Unsupported model '{model_type}'."})
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(clf, data.data, data.target, cv=5)
    return json.dumps(
        {
            "model": model_type,
            "dataset": name,
            "test_accuracy": round(acc, 4),
            "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
        }
    )


def train_pytorch_mlp(
    dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01
) -> str:
    """Train a small PyTorch MLP on the chosen dataset.

    Uses a Linear -> ReLU -> Linear net on standardized features, Adam
    for a few epochs, and returns final loss plus test accuracy.
    """
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})
    data = DATASETS[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    # Feature standardization
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    num_features = X_train.shape[1]
    num_classes = len(np.unique(data.target))
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.long)
    model = nn.Sequential(
        nn.Linear(num_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        test_out = model(X_val_t)
        test_preds = torch.argmax(test_out, dim=1)
        acc = (test_preds == y_val_t).float().mean().item()
    return json.dumps(
        {
            "framework": "PyTorch",
            "dataset": name,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "final_loss": round(float(loss.item()), 4),
            "test_accuracy": round(acc, 4),
        }
    )


def tune_hyperparameters(
    dataset_name: str, model_type: str = "svc", cv: int = 5
) -> str:
    """Run a compact grid search for SVC or DecisionTree.

    Grids are kept tiny (12 or fewer combos) so a single call finishes
    quickly on CPU. Returns the best params and best CV score as JSON.
    """
    try:
        name = dataset_name.lower().strip()
        if name not in DATASETS:
            return json.dumps({"error": f"Dataset '{name}' not found. Options: {list(DATASETS.keys())}"})
        model_type = model_type.lower().strip()
        if model_type not in ("svc", "svm", "decision_tree", "decisiontree"):
            return json.dumps(
                {"error": f"Unsupported model_type '{model_type}'. Options: ['svc', 'decision_tree']"}
            )
        # Normalize aliases
        if model_type in ("svm",):
            model_type = "svc"
        if model_type in ("decisiontree",):
            model_type = "decision_tree"

        try:
            cv = int(cv)
        except Exception:
            return json.dumps({"error": f"Invalid cv '{cv}': must be an integer >=2."})
        if cv < 2 or cv > 10:
            return json.dumps({"error": f"Invalid cv={cv}: must be between 2 and 10."})

        data = DATASETS[name]()
        X, y = data.data, data.target

        # Use scaled features for SVC (distance-based), raw for tree
        if model_type == "svc":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            estimator = SVC()
            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale"],
            }
        else:  # decision_tree
            estimator = DecisionTreeClassifier(random_state=42)
            param_grid = {
                "max_depth": [3, 5, None],
                "min_samples_split": [2, 5],
                "criterion": ["gini", "entropy"],
            }

        n_candidates = 1
        for v in param_grid.values():
            n_candidates *= len(v)

        grid = GridSearchCV(estimator, param_grid, cv=cv, scoring="accuracy", n_jobs=1)
        grid.fit(X, y)

        # Make best_params JSON-serializable (None -> None is fine)
        best_params = grid.best_params_
        # Ensure values are JSON-safe (e.g. convert numpy types)
        serializable_params = {}
        for k, v in best_params.items():
            if isinstance(v, (np.integer, np.floating)):
                serializable_params[k] = v.item()
            else:
                serializable_params[k] = v

        return json.dumps(
            {
                "model": model_type,
                "dataset": name,
                "best_params": serializable_params,
                "best_cv_score": round(float(grid.best_score_), 4),
                "cv": cv,
                "n_candidates": n_candidates,
            }
        )
    except Exception as e:
        return json.dumps({"error": f"Hyperparameter tuning failed: {str(e)}"})


def select_features(
    dataset_name: str, n_features: int = 2, cv: int = 3
) -> str:
    """Pick features with PCA and forward sequential selection.

    Standardizes the data, runs PCA for explained variance, then uses
    SequentialFeatureSelector with LogisticRegression. Returns which
    features were kept, their indices, and the CV accuracy on the subset.
    """
    try:
        name = dataset_name.lower().strip()
        if name not in DATASETS:
            return json.dumps({"error": f"Dataset '{name}' not found. Options: {list(DATASETS.keys())}"})

        try:
            n_features = int(n_features)
        except Exception:
            return json.dumps({"error": f"Invalid n_features '{n_features}': must be an integer."})
        try:
            cv = int(cv)
        except Exception:
            return json.dumps({"error": f"Invalid cv '{cv}': must be an integer >=2."})
        if cv < 2 or cv > 10:
            return json.dumps({"error": f"Invalid cv={cv}: must be between 2 and 10."})

        data = DATASETS[name]()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        n_samples, n_orig = X.shape

        if n_features < 1 or n_features > n_orig:
            return json.dumps({"error": f"Invalid n_features={n_features}: must be between 1 and {n_orig} for '{name}'."})
        if n_features > n_samples:
            return json.dumps({"error": f"Invalid n_features={n_features}: exceeds n_samples={n_samples}."})

        # Standardize (PCA and selector both benefit from scaling)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA: n_components equals n_features for comparable dimensionality
        n_components = min(n_features, n_orig, n_samples)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        explained = pca.explained_variance_ratio_.tolist()
        cumulative = float(np.sum(pca.explained_variance_ratio_))

        # Sequential forward selection
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        sfs = SequentialFeatureSelector(
            estimator,
            n_features_to_select=n_features,
            direction="forward",
            cv=cv,
            n_jobs=1,
        )
        sfs.fit(X_scaled, y)
        mask = sfs.get_support()
        selected_indices = [int(i) for i in np.where(mask)[0].tolist()]
        selected_features = [feature_names[i] for i in selected_indices]

        # CV score on selected subset
        X_selected = X_scaled[:, mask]
        cv_scores = cross_val_score(estimator, X_selected, y, cv=cv)
        selector_mean = round(float(cv_scores.mean()), 4)
        selector_std = round(float(cv_scores.std()), 4)

        return json.dumps(
            {
                "dataset": name,
                "n_samples": int(n_samples),
                "n_features_original": int(n_orig),
                "n_selected": int(n_features),
                "cv": int(cv),
                "selected_features": selected_features,
                "selected_indices": selected_indices,
                "pca_explained_variance_ratio": [round(float(v), 4) for v in explained],
                "pca_cumulative_variance": round(cumulative, 4),
                "selector_cv_mean_accuracy": selector_mean,
                "selector_cv_std": selector_std,
            }
        )
    except Exception as e:
        return json.dumps({"error": f"Feature selection failed: {str(e)}"})


class RegularizedMLP(nn.Module):
    """A deeper MLP with dropout and optional batch-norm.

    Two hidden blocks (Linear -> BatchNorm? -> ReLU -> Dropout) then a
    final linear layer. Supports step or cosine LR scheduling, but the
    net itself is just a straightforward regularized classifier.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout_rate: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        # Block 1
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        # Block 2: deep hidden
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        # Output
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_regularized_mlp(
    dataset_name: str,
    hidden_dim: int = 64,
    dropout_rate: float = 0.3,
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    use_batchnorm: bool = True,
    scheduler: str = "step",
) -> str:
    """Train a deeper MLP with dropout, batch-norm and a scheduler.

    Returns the same core fields as the plain MLP (loss, accuracy) plus
    the regularization settings and final learning rate. Bad inputs come
    back as {"error": ...} JSON instead of raising.
    """
    try:
        name = dataset_name.lower().strip()
        if name not in DATASETS:
            return json.dumps({"error": f"Dataset '{name}' not found. Options: {list(DATASETS.keys())}"})

        # ---- validate / coerce params ----
        try:
            hidden_dim = int(hidden_dim)
        except Exception:
            return json.dumps({"error": f"Invalid hidden_dim '{hidden_dim}': must be an integer."})
        if hidden_dim < 4 or hidden_dim > 512:
            return json.dumps({"error": f"Invalid hidden_dim={hidden_dim}: must be between 4 and 512."})

        try:
            dropout_rate = float(dropout_rate)
        except Exception:
            return json.dumps({"error": f"Invalid dropout_rate '{dropout_rate}': must be a float 0-0.9."})
        if not 0.0 <= dropout_rate <= 0.9:
            return json.dumps({"error": f"Invalid dropout_rate={dropout_rate}: must be between 0.0 and 0.9."})

        try:
            epochs = int(epochs)
        except Exception:
            return json.dumps({"error": f"Invalid epochs '{epochs}': must be an integer."})
        if epochs < 1 or epochs > 500:
            return json.dumps({"error": f"Invalid epochs={epochs}: must be between 1 and 500."})

        try:
            lr = float(lr)
        except Exception:
            return json.dumps({"error": f"Invalid lr '{lr}': must be a float."})
        if not 0 < lr <= 1.0:
            return json.dumps({"error": f"Invalid lr={lr}: must be between 0 and 1.0."})

        try:
            weight_decay = float(weight_decay)
        except Exception:
            return json.dumps({"error": f"Invalid weight_decay '{weight_decay}': must be a float."})
        if not 0 <= weight_decay <= 1.0:
            return json.dumps({"error": f"Invalid weight_decay={weight_decay}: must be between 0 and 1.0."})

        # use_batchnorm may arrive as string from LLM JSON ("true")
        if isinstance(use_batchnorm, str):
            low = use_batchnorm.lower().strip()
            if low in ("true", "1", "yes"):
                use_batchnorm = True
            elif low in ("false", "0", "no"):
                use_batchnorm = False
            else:
                return json.dumps({"error": f"Invalid use_batchnorm '{use_batchnorm}': must be boolean."})
        use_batchnorm = bool(use_batchnorm)

        scheduler = str(scheduler).lower().strip()
        if scheduler not in ("step", "steplr", "cosine", "cosineannealinglr"):
            return json.dumps({"error": f"Invalid scheduler '{scheduler}': options ['step', 'cosine']."})
        # normalize
        if scheduler in ("steplr",):
            scheduler = "step"
        if scheduler in ("cosineannealinglr",):
            scheduler = "cosine"

        data = DATASETS[name]()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
        )
        # Standardize
        mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        num_features = X_train.shape[1]
        num_classes = len(np.unique(data.target))

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_test, dtype=torch.float32)
        y_val_t = torch.tensor(y_test, dtype=torch.long)

        model = RegularizedMLP(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if scheduler == "step":
            step_size = max(10, epochs // 5)
            lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)
        else:  # cosine
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        model.train()
        final_loss_val = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            out = model(X_t)
            loss = criterion(out, y_t)
            # NaN/inf guard: surface as error JSON rather than crash
            if not torch.isfinite(loss):
                return json.dumps({"error": "Training diverged: loss is NaN/inf. Try lower lr or dropout_rate."})
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            final_loss_val = float(loss.item())

        model.eval()
        with torch.no_grad():
            test_out = model(X_val_t)
            test_preds = torch.argmax(test_out, dim=1)
            acc = (test_preds == y_val_t).float().mean().item()
            final_lr = float(optimizer.param_groups[0]["lr"])

        return json.dumps(
            {
                "framework": "PyTorch",
                "dataset": name,
                "hidden_dim": int(hidden_dim),
                "dropout_rate": round(float(dropout_rate), 4),
                "use_batchnorm": bool(use_batchnorm),
                "scheduler": scheduler,
                "weight_decay": float(weight_decay),
                "epochs": int(epochs),
                "lr": float(lr),
                "final_lr": round(final_lr, 6),
                "final_loss": round(float(final_loss_val), 4),
                "test_accuracy": round(float(acc), 4),
            }
        )
    except Exception as e:
        return json.dumps({"error": f"Regularized MLP training failed: {str(e)}"})


def train_pytorch_mlp_cv(
    dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01, cv: int = 5
) -> str:
    """Train the MLP with real k-fold CV plus a held-out test score.

    Uses stratified folds, standardizing per-fold, same small net and
    Adam setup as the single-split version. Returns CV mean/std, test
    accuracy and final loss.
    """
    try:
        name = dataset_name.lower().strip() if isinstance(dataset_name, str) else ""
        if name not in DATASETS:
            return json.dumps({"error": f"Dataset '{name}' not found. Options: {list(DATASETS.keys())}"})
        # ---- validate / coerce params ----
        try:
            hidden_dim = int(hidden_dim)
        except Exception:
            return json.dumps({"error": f"Invalid hidden_dim '{hidden_dim}': must be an integer."})
        if hidden_dim < 4 or hidden_dim > 512:
            return json.dumps({"error": f"Invalid hidden_dim={hidden_dim}: must be between 4 and 512."})
        try:
            epochs = int(epochs)
        except Exception:
            return json.dumps({"error": f"Invalid epochs '{epochs}': must be an integer."})
        if epochs < 1 or epochs > 500:
            return json.dumps({"error": f"Invalid epochs={epochs}: must be between 1 and 500."})
        try:
            lr = float(lr)
        except Exception:
            return json.dumps({"error": f"Invalid lr '{lr}': must be a float."})
        if not 0 < lr <= 1.0:
            return json.dumps({"error": f"Invalid lr={lr}: must be between 0 and 1.0."})
        try:
            cv = int(cv)
        except Exception:
            return json.dumps({"error": f"Invalid cv '{cv}': must be an integer."})
        if cv < 2 or cv > 10:
            return json.dumps({"error": f"Invalid cv={cv}: must be between 2 and 10."})

        data = DATASETS[name]()
        X, y = data.data, data.target
        num_features = X.shape[1]
        num_classes = len(np.unique(y))

        # ---- real k-fold CV ----
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        fold_accs: list[float] = []
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            mean, std = X_train_fold.mean(axis=0), X_train_fold.std(axis=0) + 1e-7
            X_train_s = (X_train_fold - mean) / std
            X_val_s = (X_val_fold - mean) / std

            X_t = torch.tensor(X_train_s, dtype=torch.float32)
            y_t = torch.tensor(y_train_fold, dtype=torch.long)
            X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
            y_val_t = torch.tensor(y_val_fold, dtype=torch.long)

            model = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            for _ in range(epochs):
                optimizer.zero_grad()
                out = model(X_t)
                loss = criterion(out, y_t)
                if not torch.isfinite(loss):
                    return json.dumps({"error": "Training diverged: loss is NaN/inf. Try lower lr or hidden_dim."})
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                test_out = model(X_val_t)
                test_preds = torch.argmax(test_out, dim=1)
                acc = (test_preds == y_val_t).float().mean().item()
                fold_accs.append(float(acc))

        cv_mean = round(float(np.mean(fold_accs)), 4) if fold_accs else 0.0
        cv_std = round(float(np.std(fold_accs)), 4) if fold_accs else 0.0

        # ---- final held-out test_accuracy (20% split) ----
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_test, dtype=torch.float32)
        y_val_t = torch.tensor(y_test, dtype=torch.long)

        model = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        final_loss_val = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            out = model(X_t)
            loss = criterion(out, y_t)
            if not torch.isfinite(loss):
                return json.dumps({"error": "Training diverged: loss is NaN/inf. Try lower lr or hidden_dim."})
            loss.backward()
            optimizer.step()
            final_loss_val = float(loss.item())

        with torch.no_grad():
            test_out = model(X_val_t)
            test_preds = torch.argmax(test_out, dim=1)
            test_acc = (test_preds == y_val_t).float().mean().item()

        return json.dumps(
            {
                "framework": "PyTorch",
                "dataset": name,
                "hidden_dim": int(hidden_dim),
                "epochs": int(epochs),
                "lr": float(lr),
                "cv": int(cv),
                "cv_mean_accuracy": cv_mean,
                "cv_std": cv_std,
                "test_accuracy": round(float(test_acc), 4),
                "final_loss": round(float(final_loss_val), 4),
            }
        )
    except Exception as e:
        return json.dumps({"error": f"PyTorch MLP CV training failed: {str(e)}"})


# Registry mapping tool names to callable Python functions
AVAILABLE_TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
    "tune_hyperparameters": tune_hyperparameters,
    "select_features": select_features,
    "train_regularized_mlp": train_regularized_mlp,
    "train_pytorch_mlp_cv": train_pytorch_mlp_cv,
}
