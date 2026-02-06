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
    ("XGB", "XGBoost", "Boosting", "Medium", False, True, False, False),
    ("LGB", "LightGBM", "Boosting", "Fast", False, False, True, False),
    ("CB", "CatBoost", "Boosting", "Medium", False, False, False, True),
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


def _yes_no(flag):
    # type: (bool) -> str
    """Return 'Yes' or 'No' for a boolean."""
    return "Yes" if flag else "No"


def build_comparison_data(deps):
    # type: (Dict[str, bool]) -> List[Tuple[str, str, str, str, bool, str, str, str, str]]
    """Build the table rows for the comparison panel.

    Parameters
    ----------
    deps : dict[str, bool]
        Output of :func:`~guided_workflow_widget.check_dependency_availability`.

    Returns
    -------
    list of tuples
        Each tuple: (code, name, type, speed, available, optuna, shap, smote, class_weights).
        ``available`` is a bool; the rest are strings.

    Notes
    -----
    All 12 classifiers are always represented regardless of what is
    installed.  The ``available`` flag drives row colouring in the UI.
    """
    rows = []  # type: List[Tuple[str, str, str, str, bool, str, str, str, str]]
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
        rows.append((
            code,
            name,
            algo_type,
            speed,
            available,
            _yes_no(features.get("optuna", False)),
            _yes_no(features.get("shap", False)),
            _yes_no(features.get("smote", False)),
            _yes_no(features.get("class_weights", False)),
        ))
    return rows


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

# Column indices
_COL_ALGORITHM = 0
_COL_TYPE = 1
_COL_SPEED = 2
_COL_AVAILABLE = 3
_COL_OPTUNA = 4
_COL_SHAP = 5
_COL_SMOTE = 6
_COL_CLASS_WEIGHTS = 7

_HEADERS = ["Algorithm", "Type", "Speed", "Available", "Optuna", "SHAP", "SMOTE", "Class Weights"]


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
        self.setMinimumSize(720, 400)

        layout = QVBoxLayout()

        hint = QLabel("Algorithms highlighted in red have missing dependencies.")
        layout.addWidget(hint)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.horizontalHeader().setStretchLastSection(True)
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

        for row_idx, (code, name, algo_type, speed, available, optuna, shap, smote, cw) in enumerate(rows):
            colour = black if available else red_text
            cells = [name, algo_type, speed, _yes_no(available), optuna, shap, smote, cw]
            for col_idx, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setForeground(colour)
                if not available:
                    item.setBackground(red_bg)
                item.setFlags(item.flags() & ~editable_flag)
                self.table.setItem(row_idx, col_idx, item)

        self.table.resizeColumnsToContents()

    def _on_use_selected(self):
        # type: () -> None
        """Emit algorithmSelected with the name from the current row."""
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, _COL_ALGORITHM).text()
        self.algorithmSelected.emit(name)
        self.close()


# ---------------------------------------------------------------------------
# Minimal QPushButton import needed for the dialog (not at module top to
# keep the module importable even in unusual Qt configurations)
# ---------------------------------------------------------------------------
from qgis.PyQt.QtWidgets import QPushButton  # noqa: E402
