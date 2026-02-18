"""About dialog for dzetsaka plugin."""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

# Qt5 / Qt6 enum compatibility
try:
    _AlignCenter = Qt.AlignmentFlag.AlignCenter
    _AlignLeft = Qt.AlignmentFlag.AlignLeft
    _AlignTop = Qt.AlignmentFlag.AlignTop
    _KeepAspectRatio = Qt.AspectRatioMode.KeepAspectRatio
    _SmoothTransformation = Qt.TransformationMode.SmoothTransformation
    _ButtonOk = QDialogButtonBox.StandardButton.Ok
except AttributeError:
    _AlignCenter = Qt.AlignCenter  # type: ignore[attr-defined]
    _AlignLeft = Qt.AlignLeft  # type: ignore[attr-defined]
    _AlignTop = Qt.AlignTop  # type: ignore[attr-defined]
    _KeepAspectRatio = Qt.KeepAspectRatio  # type: ignore[attr-defined]
    _SmoothTransformation = Qt.SmoothTransformation  # type: ignore[attr-defined]
    _ButtonOk = QDialogButtonBox.Ok  # type: ignore[attr-defined]


def show_about_dialog(plugin) -> None:
    """Display the dzetsaka About window."""
    dialog = QDialog(plugin.iface.mainWindow())
    dialog.setWindowTitle("About dzetsaka")
    dialog.setMinimumWidth(460)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(12)

    logo = QLabel()
    logo.setAlignment(_AlignCenter)
    pixmap = QPixmap(plugin.get_icon_path("logo.png"))
    if pixmap.isNull():
        pixmap = QPixmap(plugin.get_icon_path("icon.png"))
    if not pixmap.isNull():
        logo.setPixmap(
            pixmap.scaled(
                120,
                120,
                _KeepAspectRatio,
                _SmoothTransformation,
            ),
        )
    layout.addWidget(logo)

    version = getattr(plugin, "plugin_version", None) or plugin._read_plugin_version()

    title = QLabel(f"<h2>dzetsaka {version}</h2>")
    title.setAlignment(_AlignCenter)
    layout.addWidget(title)

    about = QLabel(
        "dzetsaka is an AI-powered remote sensing classification plugin for QGIS. "
        "It provides modern machine learning workflows, advanced optimization, "
        "guided UI/UX, reusable recipes, and rich report generation.",
    )
    about.setWordWrap(True)
    about.setAlignment(_AlignLeft | _AlignTop)
    layout.addWidget(about)

    author = QLabel(
        "Nicolas Karasiak, data scientist, researcher.<br>"
        "Available on X: <a href='https://x.com/nkarasiak'>@nkarasiak</a><br>"
        "LinkedIn: <a href='https://www.linkedin.com/in/nicolas-karasiak/'>nicolas-karasiak</a>",
    )
    author.setOpenExternalLinks(True)
    author.setWordWrap(True)
    layout.addWidget(author)

    buttons = QDialogButtonBox(_ButtonOk)
    buttons.accepted.connect(dialog.accept)
    layout.addWidget(buttons)

    try:
        dialog.exec_()
    except AttributeError:
        dialog.exec()
