"""Main dock runtime helpers for the QGIS plugin UI."""

from __future__ import annotations


def _on_changed_layer(gui) -> None:
    """Update class field options when selected vector layer changes."""
    dock = gui.dock_widget
    dock.inField.clear()
    if (
        dock.inField.currentText() == ""
        and dock.inShape.currentLayer()
        and dock.inShape.currentLayer() != "NoneType"
    ):
        try:
            active_layer = dock.inShape.currentLayer()
            provider = active_layer.dataProvider()
            fields = provider.fields()
            field_names = [field.name() for field in fields]
            dock.inField.addItems(field_names)
        except BaseException:
            gui.log.warning("dzetsaka cannot change active layer. Maybe you opened an OSM/Online background ?")


def _update_optional_title(gui) -> None:
    dock = gui.dock_widget
    title = "Optional ▼" if dock.mGroupBox.isCollapsed() else "Optional ▲"
    dock.mGroupBox.setTitle(title)


def resize_dock(gui) -> None:
    """Resize dock widget based on group box collapse state."""
    dock = gui.dock_widget
    if dock.mGroupBox.isCollapsed():
        dock.mGroupBox.setFixedHeight(20)
        dock.setFixedHeight(390)
    else:
        dock.setMinimumHeight(520)
        dock.mGroupBox.setMinimumHeight(160)


def run_plugin_ui(gui, *, dock_area, ui_module) -> None:
    """Run method that loads and starts the main plugin dock."""
    if not gui.pluginIsActive or gui.dock_widget is None:
        gui.pluginIsActive = True

        if gui.dock_widget is None:
            gui.dock_widget = ui_module.dzetsakaDockWidget()

        dock = gui.dock_widget
        dock.closingPlugin.connect(gui.onClosePlugin)

        from qgis.core import QgsProviderRegistry

        except_raster = QgsProviderRegistry.instance().providerList()
        except_raster.remove("gdal")
        dock.inRaster.setExcludedProviders(except_raster)

        except_vector = QgsProviderRegistry.instance().providerList()
        except_vector.remove("ogr")
        dock.inShape.setExcludedProviders(except_vector)

        dock.outRaster.clear()
        dock.outRasterButton.clicked.connect(gui.select_output_file)

        dock.outModel.clear()
        dock.checkOutModel.clicked.connect(gui.checkbox_state)

        dock.inModel.clear()
        dock.checkInModel.clicked.connect(gui.checkbox_state)

        dock.maskPathEdit.clear()
        dock.checkInMask.clicked.connect(gui.checkbox_state)

        dock.confusionMatrixPathEdit.clear()
        dock.checkOutMatrix.clicked.connect(gui.checkbox_state)

        dock.confidenceMapPathEdit.clear()
        dock.confidenceMapCheckBox.clicked.connect(gui.checkbox_state)

        dock.inField.clear()

        if hasattr(dock, "messageBanner"):
            dock.messageBanner.linkActivated.connect(lambda _=None: gui.open_dashboard())

        gui.iface.addDockWidget(dock_area, dock)
        dock.show()

        _on_changed_layer(gui)
        dock.inShape.currentIndexChanged[int].connect(lambda _index: _on_changed_layer(gui))

        dock.settingsButton.clicked.connect(gui.open_dashboard)
        dock.runClassificationButton.clicked.connect(gui.run_classification)

        dock.mGroupBox.setSaveCollapsedState(False)
        dock.mGroupBox.setCollapsed(True)
        resize_dock(gui)
        _update_optional_title(gui)
        dock.mGroupBox.collapsedStateChanged.connect(gui.resizeDock)
        dock.mGroupBox.collapsedStateChanged.connect(lambda _state: _update_optional_title(gui))



