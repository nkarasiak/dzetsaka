"""About dialog for dzetsaka plugin."""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


def show_about_dialog(plugin) -> None:
    """Display the dzetsaka About window."""
    dialog = QDialog(plugin.iface.mainWindow())
    dialog.setWindowTitle("About dzetsaka")
    dialog.setMinimumWidth(460)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(12)

    logo = QLabel()
    logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
    pixmap = QPixmap(plugin.get_icon_path("logo.png"))
    if pixmap.isNull():
        pixmap = QPixmap(plugin.get_icon_path("icon.png"))
    if not pixmap.isNull():
        logo.setPixmap(
            pixmap.scaled(
                120,
                120,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
    layout.addWidget(logo)

    version = getattr(plugin, "plugin_version", None) or plugin._read_plugin_version()

    title = QLabel(f"<h2>dzetsaka {version}</h2>")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(title)

    about = QLabel(
        "dzetsaka is an AI-powered remote sensing classification plugin for QGIS. "
        "It provides modern machine learning workflows, advanced optimization, "
        "guided UI/UX, reusable recipes, and rich report generation."
    )
    about.setWordWrap(True)
    about.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    layout.addWidget(about)

    author = QLabel(
        "Nicolas Karasiak, data scientist, researcher.<br>"
        "Available on X: <a href='https://x.com/nkarasiak'>@nkarasiak</a><br>"
        "LinkedIn: <a href='https://www.linkedin.com/in/nicolas-karasiak/'>nicolas-karasiak</a>"
    )
    author.setOpenExternalLinks(True)
    author.setWordWrap(True)
    layout.addWidget(author)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
    buttons.accepted.connect(dialog.accept)
    layout.addWidget(buttons)

    dialog.exec()
