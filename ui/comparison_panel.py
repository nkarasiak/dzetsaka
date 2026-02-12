"""Algorithm comparison panel for dzetsaka.

A modal QDialog that displays all 12 supported classifiers in a
colour-coded table.  Rows whose hard dependencies are missing are
shown in red.  The user can select an algorithm and click
"Use Selected" to propagate the choice back to the guided workflow's
Input & Algorithm page via the ``algorithmSelected`` signal.

The table data is built by ``build_comparison_data``, a module-level
function that can be tested without a Qt runtime.

Author:
    Nicolas Karasiak
"""

from typing import Dict, List, Tuple

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QAbstractItemView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QPushButton,
    QTextEdit,
)

from .dashboard_widget import check_dependency_availability

# ---------------------------------------------------------------------------
# Static table data
# ---------------------------------------------------------------------------

# Each entry: (code, name, classifier_type, speed, needs_sklearn, needs_xgboost, needs_lightgbm, needs_catboost)
_ALGO_TABLE_DATA = [
    ("GMM", "Gaussian Mixture Model", "Probabilistic", "Fast", False, False, False, False),
    ("RF", "Random Forest", "Ensemble", "Fast", True, False, False, False),
    ("SVM", "Support Vector Machine", "Kernel", "Medium", True, False, False, False),
    ("KNN", "K-Nearest Neighbors", "Instance-based", "Medium", True, False, False, False),
    ("XGB", "XGBoost", "Boosting", "Medium", True, True, False, False),
    ("LGB", "LightGBM", "Boosting", "Fast", True, False, True, False),
    ("CB", "CatBoost", "Boosting", "Medium", True, False, False, True),
    ("ET", "Extra Trees", "Ensemble", "Fast", True, False, False, False),
    ("GBC", "Gradient Boosting Classifier", "Boosting", "Medium", True, False, False, False),
    ("LR", "Logistic Regression", "Linear", "Fast", True, False, False, False),
    ("NB", "Gaussian Naive Bayes", "Probabilistic", "Fast", True, False, False, False),
    ("MLP", "Multi-layer Perceptron", "Neural Network", "Slow", True, False, False, False),
]

# Which features each algorithm supports (optuna, shap, smote, class_weights)
_FEATURE_SUPPORT = {
    "GMM": {"optuna": False, "shap": False, "smote": False, "class_weights": False},
    "RF": {"optuna": True, "shap": True, "smote": True, "class_weights": True},
    "SVM": {"optuna": True, "shap": True, "smote": True, "class_weights": True},
    "KNN": {"optuna": True, "shap": True, "smote": True, "class_weights": False},
    "XGB": {"optuna": True, "shap": True, "smote": True, "class_weights": True},
    "LGB": {"optuna": True, "shap": True, "smote": True, "class_weights": True},
    "CB": {"optuna": True, "shap": True, "smote": True, "class_weights": True},
    "ET": {"optuna": True, "shap": True, "smote": True, "class_weights": True},
    "GBC": {"optuna": True, "shap": True, "smote": True, "class_weights": True},
    "LR": {"optuna": True, "shap": True, "smote": True, "class_weights": True},
    "NB": {"optuna": False, "shap": True, "smote": True, "class_weights": False},
    "MLP": {"optuna": True, "shap": True, "smote": True, "class_weights": False},
}

# Use-case recommendations for each algorithm
_RECOMMENDATIONS = {
    "GMM": "Quick baseline, no dependencies",
    "RF": "Balanced speed/accuracy, robust to overfitting",
    "SVM": "High accuracy on small datasets",
    "KNN": "Simple, interpretable, good for irregular boundaries",
    "XGB": "State-of-the-art, large datasets, best overall accuracy",
    "LGB": "Fastest gradient boosting, large datasets",
    "CB": "Best default parameters, minimal tuning needed",
    "ET": "Fast, high variance data, parallel training",
    "GBC": "Smooth probability estimates, calibrated confidence",
    "LR": "Linear separable classes, fast inference",
    "NB": "Fast, probabilistic, works with small data",
    "MLP": "Complex non-linear patterns, neural network",
}

# Detailed explanations for "Why This?" dialogs
_ALGORITHM_EXPLANATIONS = {
    "GMM": {
        "description": "Gaussian Mixture Model is a probabilistic model that assumes data points are generated from a mixture of Gaussian distributions. It's the only algorithm with no external dependencies.",
        "when_to_use": [
            "When you need a quick baseline without installing dependencies",
            "When classes have naturally clustered distributions",
            "When you need probability estimates for classification",
            "For initial exploratory analysis",
        ],
        "tradeoffs": {
            "Accuracy": "Low-Medium (suitable for simple datasets)",
            "Speed": "Fast",
            "Memory": "Low",
            "Tuning Required": "Minimal",
        },
        "advanced_features": {
            "Optuna": False,
            "SHAP": False,
            "SMOTE": False,
            "Class Weights": False,
        },
        "examples": [
            "Quick land cover classification on simple landscapes",
            "Testing workflow before installing full dependencies",
            "Educational demonstrations of probabilistic classification",
        ],
    },
    "RF": {
        "description": "Random Forest builds multiple decision trees and combines their predictions through voting. It's one of the most robust and widely-used algorithms in remote sensing.",
        "when_to_use": [
            "When you want balanced speed and accuracy",
            "When you have limited time for parameter tuning",
            "When your dataset has high-dimensional features",
            "When you need robust handling of overfitting",
        ],
        "tradeoffs": {
            "Accuracy": "High (excellent general-purpose classifier)",
            "Speed": "Fast (parallel training)",
            "Memory": "Medium-High (stores multiple trees)",
            "Tuning Required": "Low (good default parameters)",
        },
        "advanced_features": {
            "Optuna": True,
            "SHAP": True,
            "SMOTE": True,
            "Class Weights": True,
        },
        "examples": [
            "Multi-class land cover mapping with Sentinel-2",
            "Forest type classification from hyperspectral data",
            "Agricultural crop type mapping with temporal features",
        ],
    },
    "SVM": {
        "description": "Support Vector Machine finds the optimal hyperplane that maximizes the margin between classes. Particularly effective for high-dimensional data with clear margins of separation.",
        "when_to_use": [
            "When you have small to medium-sized datasets",
            "When you need high accuracy and can afford longer training",
            "When classes are well-separated in feature space",
            "When working with high-dimensional spectral data",
        ],
        "tradeoffs": {
            "Accuracy": "Very High (excellent for small datasets)",
            "Speed": "Medium (slower training, fast prediction)",
            "Memory": "Medium",
            "Tuning Required": "High (kernel and regularization parameters)",
        },
        "advanced_features": {
            "Optuna": True,
            "SHAP": True,
            "SMOTE": True,
            "Class Weights": True,
        },
        "examples": [
            "High-resolution urban classification with limited training samples",
            "Hyperspectral image classification",
            "Binary classification tasks (water vs. non-water)",
        ],
    },
    "KNN": {
        "description": "K-Nearest Neighbors is a simple instance-based algorithm that classifies points based on the majority vote of their k nearest neighbors. Highly interpretable and requires no training phase.",
        "when_to_use": [
            "When you need a simple, interpretable method",
            "When classes have irregular boundaries",
            "When you have sufficient memory for the training set",
            "For quick prototyping and baseline comparisons",
        ],
        "tradeoffs": {
            "Accuracy": "Medium-High (depends on k and distance metric)",
            "Speed": "Medium (no training, but slow prediction on large datasets)",
            "Memory": "High (stores entire training set)",
            "Tuning Required": "Low (mainly k parameter)",
        },
        "advanced_features": {
            "Optuna": True,
            "SHAP": True,
            "SMOTE": True,
            "Class Weights": False,
        },
        "examples": [
            "Land cover classification with complex class boundaries",
            "Small dataset classification where training time isn't critical",
            "Texture-based classification in high-resolution imagery",
        ],
    },
    "XGB": {
        "description": "XGBoost (Extreme Gradient Boosting) is a highly optimized gradient boosting implementation that often achieves state-of-the-art results. It builds trees sequentially, with each tree correcting errors of previous ones.",
        "when_to_use": [
            "When you need the highest possible accuracy",
            "When you have large datasets with many features",
            "When you can invest time in hyperparameter tuning",
            "For competitions or critical applications",
        ],
        "tradeoffs": {
            "Accuracy": "Excellent (often best-in-class)",
            "Speed": "Medium (faster than traditional boosting)",
            "Memory": "Medium",
            "Tuning Required": "High (many hyperparameters to optimize)",
        },
        "advanced_features": {
            "Optuna": True,
            "SHAP": True,
            "SMOTE": True,
            "Class Weights": True,
        },
        "examples": [
            "Large-scale land cover mapping projects",
            "Multi-temporal crop classification with complex patterns",
            "Urban sprawl detection with diverse spectral-spatial features",
        ],
    },
    "LGB": {
        "description": "LightGBM is a gradient boosting framework that uses tree-based learning with leaf-wise growth. It's optimized for speed and memory efficiency, making it ideal for large datasets.",
        "when_to_use": [
            "When you have very large datasets (>100k samples)",
            "When training time is a constraint",
            "When memory usage is a concern",
            "When you need fast iteration for experimentation",
        ],
        "tradeoffs": {
            "Accuracy": "Very High (comparable to XGBoost)",
            "Speed": "Very Fast (fastest gradient boosting)",
            "Memory": "Low (efficient histogram-based algorithm)",
            "Tuning Required": "Medium (fewer parameters than XGBoost)",
        },
        "advanced_features": {
            "Optuna": True,
            "SHAP": True,
            "SMOTE": True,
            "Class Weights": True,
        },
        "examples": [
            "Continental-scale land cover mapping",
            "Real-time classification applications",
            "Time-series analysis with many temporal features",
        ],
    },
    "CB": {
        "description": "CatBoost (Categorical Boosting) is a gradient boosting algorithm that handles categorical features natively and has excellent default parameters, requiring minimal tuning.",
        "when_to_use": [
            "When you want excellent results with minimal tuning",
            "When your dataset has categorical features",
            "When you need robust handling of overfitting",
            "When you're new to gradient boosting",
        ],
        "tradeoffs": {
            "Accuracy": "Very High (excellent out-of-the-box)",
            "Speed": "Medium (similar to XGBoost)",
            "Memory": "Medium",
            "Tuning Required": "Low (best default parameters)",
        },
        "advanced_features": {
            "Optuna": True,
            "SHAP": True,
            "SMOTE": True,
            "Class Weights": True,
        },
        "examples": [
            "Land cover classification with ancillary categorical data",
            "Mixed feature classification (spectral + terrain + metadata)",
            "Production systems requiring reliable default performance",
        ],
    },
    "ET": {
        "description": "Extra Trees (Extremely Randomized Trees) is similar to Random Forest but introduces additional randomness by using random thresholds for splits, leading to faster training and reduced variance.",
        "when_to_use": [
            "When you need faster training than Random Forest",
            "When your data has high variance",
            "When you can leverage parallel processing",
            "When you want to reduce overfitting",
        ],
        "tradeoffs": {
            "Accuracy": "High (similar to Random Forest)",
            "Speed": "Very Fast (faster than RF due to random splits)",
            "Memory": "Medium-High (stores multiple trees)",
            "Tuning Required": "Low (similar to Random Forest)",
        },
        "advanced_features": {
            "Optuna": True,
            "SHAP": True,
            "SMOTE": True,
            "Class Weights": True,
        },
        "examples": [
            "Large-scale forest mapping with many spectral bands",
            "Quick exploratory classification on high-variance data",
            "Ensemble building with other tree-based methods",
        ],
    },
    "GBC": {
        "description": "Gradient Boosting Classifier builds trees sequentially in a gradient descent framework, optimizing a loss function. It produces well-calibrated probability estimates.",
        "when_to_use": [
            "When you need well-calibrated probability estimates",
            "When you want smooth decision boundaries",
            "When you need confidence scores for predictions",
            "For smaller datasets where training time is acceptable",
        ],
        "tradeoffs": {
            "Accuracy": "High (excellent for calibrated probabilities)",
            "Speed": "Medium-Slow (sequential tree building)",
            "Memory": "Medium",
            "Tuning Required": "Medium (learning rate, tree depth, etc.)",
        },
        "advanced_features": {
            "Optuna": True,
            "SHAP": True,
            "SMOTE": True,
            "Class Weights": True,
        },
        "examples": [
            "Applications requiring confidence thresholds",
            "Change detection where probability estimates matter",
            "Classification with uncertainty quantification",
        ],
    },
    "LR": {
        "description": "Logistic Regression is a linear model that predicts class probabilities using a logistic function. Fast and interpretable, suitable for linearly separable classes.",
        "when_to_use": [
            "When classes are linearly separable",
            "When you need very fast training and prediction",
            "When model interpretability is critical",
            "For baseline comparisons",
        ],
        "tradeoffs": {
            "Accuracy": "Medium (limited to linear decision boundaries)",
            "Speed": "Very Fast (fastest training and prediction)",
            "Memory": "Very Low",
            "Tuning Required": "Low (mainly regularization strength)",
        },
        "advanced_features": {
            "Optuna": True,
            "SHAP": True,
            "SMOTE": True,
            "Class_weights": True,
        },
        "examples": [
            "Binary water/land classification",
            "Simple vegetation vs. non-vegetation mapping",
            "Quick baseline before trying complex models",
        ],
    },
    "NB": {
        "description": "Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem with strong independence assumptions. Fast and works well with small datasets.",
        "when_to_use": [
            "When you have very small training datasets",
            "When you need fast training and prediction",
            "When features are relatively independent",
            "For probabilistic classification",
        ],
        "tradeoffs": {
            "Accuracy": "Medium (limited by independence assumption)",
            "Speed": "Very Fast",
            "Memory": "Very Low",
            "Tuning Required": "Minimal (few hyperparameters)",
        },
        "advanced_features": {
            "Optuna": False,
            "SHAP": True,
            "SMOTE": True,
            "Class Weights": False,
        },
        "examples": [
            "Quick classification with limited training data",
            "Text-like classification of spectral signatures",
            "Initial exploratory analysis",
        ],
    },
    "MLP": {
        "description": "Multi-layer Perceptron is a feedforward neural network with multiple hidden layers. Capable of learning complex non-linear patterns but requires more data and tuning.",
        "when_to_use": [
            "When you have complex non-linear patterns",
            "When you have sufficient training data",
            "When you can invest time in architecture tuning",
            "For deep feature learning",
        ],
        "tradeoffs": {
            "Accuracy": "High (excellent for complex patterns)",
            "Speed": "Slow (iterative training)",
            "Memory": "Medium-High (depends on architecture)",
            "Tuning Required": "High (layers, neurons, learning rate, etc.)",
        },
        "advanced_features": {
            "Optuna": True,
            "SHAP": True,
            "SMOTE": True,
            "Class Weights": False,
        },
        "examples": [
            "Hyperspectral classification with complex spectral signatures",
            "Urban scene classification with deep features",
            "Multi-modal data fusion (SAR + optical)",
        ],
    },
}


def _yes_no(flag):
    # type: (bool) -> str
    """Return 'Yes' or 'No' for a boolean."""
    return "Yes" if flag else "No"


def build_comparison_data(deps):
    # type: (Dict[str, bool]) -> List[Tuple[str, str, str, str, str, bool, str, str, str, str]]
    """Build the table rows for the comparison panel.

    Parameters
    ----------
    deps : dict[str, bool]
        Output of :func:`~classification_workflow_ui.check_dependency_availability`.

    Returns
    -------
    list of tuples
        Each tuple: (code, name, type, speed, recommended_for, available, optuna, shap, smote, class_weights).
        ``available`` is a bool; the rest are strings.

    Notes
    -----
    All 12 classifiers are always represented regardless of what is
    installed.  The ``available`` flag drives row colouring in the UI.
    """
    rows = []  # type: List[Tuple[str, str, str, str, str, bool, str, str, str, str]]
    for code, name, algo_type, speed, needs_sk, needs_xgb, needs_lgb, needs_cb in _ALGO_TABLE_DATA:
        available = True
        if needs_sk and not deps.get("sklearn", False):
            available = False
        if needs_xgb and not deps.get("xgboost", False):
            available = False
        if needs_lgb and not deps.get("lightgbm", False):
            available = False
        if needs_cb and not deps.get("catboost", False):
            available = False

        features = _FEATURE_SUPPORT.get(code, {})
        recommended_for = _RECOMMENDATIONS.get(code, "")
        rows.append((
            code,
            name,
            algo_type,
            speed,
            recommended_for,
            available,
            _yes_no(features.get("optuna", False)),
            _yes_no(features.get("shap", False)),
            _yes_no(features.get("smote", False)),
            _yes_no(features.get("class_weights", False)),
        ))
    return rows


# ---------------------------------------------------------------------------
# Explanation Dialog
# ---------------------------------------------------------------------------


class AlgorithmExplanationDialog(QDialog):
    """Modal dialog showing detailed explanation for a specific algorithm.

    Displays comprehensive information about when to use the algorithm,
    trade-offs, advanced features support, and example use cases from
    remote sensing applications.
    """

    def __init__(self, algorithm_code, parent=None):
        # type: (str, QDialog) -> None
        """Initialize the explanation dialog for a specific algorithm.

        Parameters
        ----------
        algorithm_code : str
            The algorithm code (e.g., "RF", "XGB", "GMM").
        parent : QDialog, optional
            Parent widget.
        """
        super(AlgorithmExplanationDialog, self).__init__(parent)

        # Get algorithm name
        algo_name = ""
        for code, name, _, _, _, _, _, _ in _ALGO_TABLE_DATA:
            if code == algorithm_code:
                algo_name = name
                break

        self.setWindowTitle(f"Why {algo_name}?")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout()

        # Get explanation data
        explanation = _ALGORITHM_EXPLANATIONS.get(algorithm_code, {})

        # Create scrollable text area
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        # Build HTML content
        html_content = f"<h2>{algo_name}</h2>"

        # Description
        description = explanation.get("description", "")
        if description:
            html_content += f"<h3>Description</h3><p>{description}</p>"

        # When to Use
        when_to_use = explanation.get("when_to_use", [])
        if when_to_use:
            html_content += "<h3>When to Use</h3><ul>"
            for item in when_to_use:
                html_content += f"<li>{item}</li>"
            html_content += "</ul>"

        # Trade-offs
        tradeoffs = explanation.get("tradeoffs", {})
        if tradeoffs:
            html_content += "<h3>Trade-offs</h3><table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
            for key, value in tradeoffs.items():
                html_content += f"<tr><td><b>{key}</b></td><td>{value}</td></tr>"
            html_content += "</table>"

        # Advanced Features
        advanced_features = explanation.get("advanced_features", {})
        if advanced_features:
            html_content += "<h3>Advanced Features Support</h3><ul>"
            for feature, supported in advanced_features.items():
                status = "Yes" if supported else "No"
                html_content += f"<li><b>{feature}</b>: {status}</li>"
            html_content += "</ul>"

        # Examples
        examples = explanation.get("examples", [])
        if examples:
            html_content += "<h3>Example Use Cases (Remote Sensing)</h3><ul>"
            for example in examples:
                html_content += f"<li>{example}</li>"
            html_content += "</ul>"

        text_edit.setHtml(html_content)
        layout.addWidget(text_edit)

        # OK button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

        self.setLayout(layout)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

# Column indices
_COL_ALGORITHM = 0
_COL_TYPE = 1
_COL_SPEED = 2
_COL_RECOMMENDED = 3
_COL_AVAILABLE = 4
_COL_OPTUNA = 5
_COL_SHAP = 6
_COL_SMOTE = 7
_COL_CLASS_WEIGHTS = 8
_COL_WHY_THIS = 9

_HEADERS = ["Algorithm", "Type", "Speed", "Recommended For", "Available", "Optuna", "SHAP", "SMOTE", "Class Weights", ""]


class AlgorithmComparisonPanel(QDialog):
    """Modal dialog showing a comparison table of all 12 classifiers.

    Signals
    -------
    algorithmSelected : str
        Emitted with the full name of the algorithm when the user clicks
        "Use Selected".
    """

    algorithmSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        """Initialise AlgorithmComparisonPanel and populate the table."""
        super(AlgorithmComparisonPanel, self).__init__(parent)
        self.setWindowTitle("Algorithm Comparison")
        self.setMinimumSize(900, 500)

        layout = QVBoxLayout()

        hint = QLabel("Algorithms highlighted in red have missing dependencies. Click 'Why This?' for detailed information.")
        layout.addWidget(hint)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        # Qt6 compatibility: SelectRows lives on QAbstractItemView, not QTableWidget
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # Qt6 compatibility: NoEditTriggers lives on QAbstractItemView
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        self._populate_table()

        # Buttons
        btn_layout = QHBoxLayout()
        self.useBtn = QPushButton("Use Selected")
        self.useBtn.clicked.connect(self._on_use_selected)
        btn_layout.addStretch()
        btn_layout.addWidget(self.useBtn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _populate_table(self):
        # type: () -> None
        """Fill the QTableWidget with data from build_comparison_data."""
        deps = check_dependency_availability()
        rows = build_comparison_data(deps)
        self.table.setRowCount(len(rows))

        red_text = QColor(120, 0, 0)
        black = QColor(0, 0, 0)
        red_bg = QColor(255, 235, 235)

        try:
            editable_flag = Qt.ItemIsEditable
        except AttributeError:
            editable_flag = Qt.ItemFlag.ItemIsEditable

        for row_idx, (code, name, algo_type, speed, recommended, available, optuna, shap, smote, cw) in enumerate(rows):
            colour = black if available else red_text
            cells = [name, algo_type, speed, recommended, _yes_no(available), optuna, shap, smote, cw]
            for col_idx, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setForeground(colour)
                if not available:
                    item.setBackground(red_bg)
                item.setFlags(item.flags() & ~editable_flag)
                self.table.setItem(row_idx, col_idx, item)

            # Add "Why This?" button in the last column
            why_btn = QPushButton("Why This?")
            why_btn.setMaximumWidth(100)
            why_btn.setProperty("algorithm_code", code)
            why_btn.clicked.connect(self._on_why_this_clicked)
            self.table.setCellWidget(row_idx, _COL_WHY_THIS, why_btn)

        self.table.resizeColumnsToContents()
        # Make the "Recommended For" column stretch to use available space
        try:
            from qgis.PyQt.QtWidgets import QHeaderView
            resize_mode = QHeaderView.ResizeMode.Stretch
        except (ImportError, AttributeError):
            # Qt5 compatibility
            from qgis.PyQt.QtWidgets import QHeaderView
            resize_mode = QHeaderView.Stretch

        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(_COL_RECOMMENDED, resize_mode)

    def _on_use_selected(self):
        # type: () -> None
        """Emit algorithmSelected with the name from the current row."""
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, _COL_ALGORITHM).text()
        self.algorithmSelected.emit(name)
        self.close()

    def _on_why_this_clicked(self):
        # type: () -> None
        """Open the explanation dialog for the clicked algorithm."""
        sender = self.sender()
        if sender is None:
            return
        algorithm_code = sender.property("algorithm_code")
        if algorithm_code:
            dialog = AlgorithmExplanationDialog(algorithm_code, self)
            dialog.exec()

