"""Main dock runtime helpers for the QGIS plugin UI."""

from __future__ import annotations


def _on_changed_layer(gui) -> None:
    """Update class field options when selected vector layer changes."""
    gui.dockwidget.inField.clear()
    if (
        gui.dockwidget.inField.currentText() == ""
        and gui.dockwidget.inShape.currentLayer()
        and gui.dockwidget.inShape.currentLayer() != "NoneType"
    ):
        try:
            active_layer = gui.dockwidget.inShape.currentLayer()
            provider = active_layer.dataProvider()
            fields = provider.fields()
            field_names = [field.name() for field in fields]
            gui.dockwidget.inField.addItems(field_names)
        except BaseException:
            gui.log.warning("dzetsaka cannot change active layer. Maybe you opened an OSM/Online background ?")


def _update_optional_title(gui) -> None:
    title = "Optional ▼" if gui.dockwidget.mGroupBox.isCollapsed() else "Optional ▲"
    gui.dockwidget.mGroupBox.setTitle(title)


def resize_dock(gui) -> None:
    """Resize dock widget based on group box collapse state."""
    if gui.dockwidget.mGroupBox.isCollapsed():
        gui.dockwidget.mGroupBox.setFixedHeight(20)
        gui.dockwidget.setFixedHeight(390)
    else:
        gui.dockwidget.setMinimumHeight(520)
        gui.dockwidget.mGroupBox.setMinimumHeight(160)


def run_plugin_ui(gui, *, dock_area, ui_module) -> None:
    """Run method that loads and starts the main plugin dock."""
    if not gui.pluginIsActive or gui.dockwidget is None:
        gui.pluginIsActive = True

        if gui.dockwidget is None:
            gui.dockwidget = ui_module.dzetsakaDockWidget()

        gui.dockwidget.closingPlugin.connect(gui.onClosePlugin)

        from qgis.core import QgsProviderRegistry

        except_raster = QgsProviderRegistry.instance().providerList()
        except_raster.remove("gdal")
        gui.dockwidget.inRaster.setExcludedProviders(except_raster)

        except_vector = QgsProviderRegistry.instance().providerList()
        except_vector.remove("ogr")
        gui.dockwidget.inShape.setExcludedProviders(except_vector)

        gui.dockwidget.outRaster.clear()
        gui.dockwidget.outRasterButton.clicked.connect(gui.select_output_file)

        gui.dockwidget.outModel.clear()
        gui.dockwidget.checkOutModel.clicked.connect(gui.checkbox_state)

        gui.dockwidget.inModel.clear()
        gui.dockwidget.checkInModel.clicked.connect(gui.checkbox_state)

        gui.dockwidget.inMask.clear()
        gui.dockwidget.checkInMask.clicked.connect(gui.checkbox_state)

        gui.dockwidget.outMatrix.clear()
        gui.dockwidget.checkOutMatrix.clicked.connect(gui.checkbox_state)

        gui.dockwidget.outConfidenceMap.clear()
        gui.dockwidget.checkInConfidence.clicked.connect(gui.checkbox_state)

        gui.dockwidget.inField.clear()

        if hasattr(gui.dockwidget, "messageBanner"):
            gui.dockwidget.messageBanner.linkActivated.connect(lambda _=None: gui.run_wizard())

        gui.iface.addDockWidget(dock_area, gui.dockwidget)
        gui.dockwidget.show()

        _on_changed_layer(gui)
        gui.dockwidget.inShape.currentIndexChanged[int].connect(lambda _index: _on_changed_layer(gui))

        gui.dockwidget.settingsButton.clicked.connect(gui.run_wizard)
        gui.dockwidget.performMagic.clicked.connect(gui.runMagic)

        gui.dockwidget.mGroupBox.setSaveCollapsedState(False)
        gui.dockwidget.mGroupBox.setCollapsed(True)
        resize_dock(gui)
        _update_optional_title(gui)
        gui.dockwidget.mGroupBox.collapsedStateChanged.connect(gui.resizeDock)
        gui.dockwidget.mGroupBox.collapsedStateChanged.connect(lambda _state: _update_optional_title(gui))
