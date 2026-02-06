"""File chooser and checkbox handlers for the main dock form."""

from __future__ import annotations

import os

from qgis.PyQt.QtWidgets import QFileDialog


def select_output_file(gui) -> None:
    """Select output file and enforce expected file extension."""
    sender = gui.sender()

    file_name, _filter = QFileDialog.getSaveFileName(
        gui.dockwidget, "Select output file", gui.lastSaveDir, "TIF (*.tif)"
    )
    gui.rememberLastSaveDir(file_name)

    if not file_name:
        return

    file_name, file_extension = os.path.splitext(file_name)

    if sender == gui.dockwidget.outRasterButton:
        if file_extension != ".tif":
            gui.dockwidget.outRaster.setText(file_name + ".tif")
        else:
            gui.dockwidget.outRaster.setText(file_name + file_extension)

    if "self.historicalmap" in locals():
        if sender == gui.historicalmap.outRasterButton:
            if file_extension != ".tif":
                gui.historicalmap.outRaster.setText(file_name + ".tif")
            else:
                gui.historicalmap.outRaster.setText(file_name + file_extension)
        if sender == gui.historicalmap.outShpButton:
            if file_extension != ".shp":
                gui.historicalmap.outShp.setText(file_name + ".shp")
            else:
                gui.historicalmap.outShp.setText(file_name + file_extension)

    if "self.filters_dock" in locals():
        if sender == gui.filters_dock.outRasterButton:
            if file_extension != ".tif":
                gui.filters_dock.outRaster.setText(file_name + ".tif")
        else:
            gui.filters_dock.outRaster.setText(file_name + file_extension)


def checkbox_state(gui) -> None:
    """Handle checkbox-driven file inputs/outputs in the main dock."""
    sender = gui.sender()

    if sender == gui.dockwidget.checkInModel and gui.dockwidget.checkInModel.isChecked():
        file_name, _filter = QFileDialog.getOpenFileName(gui.dockwidget, "Select your file", gui.lastSaveDir)
        gui.rememberLastSaveDir(file_name)
        if file_name != "":
            gui.dockwidget.inModel.setText(file_name)
            gui.dockwidget.inModel.setEnabled(True)
            gui.dockwidget.inShape.setEnabled(False)
            gui.dockwidget.inField.setEnabled(False)
        else:
            gui.dockwidget.checkInModel.setChecked(False)
            gui.dockwidget.inModel.setEnabled(False)
            gui.dockwidget.inShape.setEnabled(True)
            gui.dockwidget.inField.setEnabled(True)
    elif sender == gui.dockwidget.checkInModel:
        gui.dockwidget.inModel.clear()
        gui.dockwidget.inModel.setEnabled(False)
        gui.dockwidget.inShape.setEnabled(True)
        gui.dockwidget.inField.setEnabled(True)

    if sender == gui.dockwidget.checkOutModel and gui.dockwidget.checkOutModel.isChecked():
        file_name, _filter = QFileDialog.getSaveFileName(gui.dockwidget, "Select output file", gui.lastSaveDir)
        gui.rememberLastSaveDir(file_name)
        if file_name != "":
            gui.dockwidget.outModel.setText(file_name)
            gui.dockwidget.outModel.setEnabled(True)
        else:
            gui.dockwidget.checkOutModel.setChecked(False)
            gui.dockwidget.outModel.setEnabled(False)
    elif sender == gui.dockwidget.checkOutModel:
        gui.dockwidget.outModel.clear()
        gui.dockwidget.outModel.setEnabled(False)

    if sender == gui.dockwidget.checkInMask and gui.dockwidget.checkInMask.isChecked():
        file_name, _filter = QFileDialog.getOpenFileName(
            gui.dockwidget,
            "Select your mask raster",
            gui.lastSaveDir,
            "TIF (*.tif)",
        )
        gui.rememberLastSaveDir(file_name)
        if file_name != "":
            gui.dockwidget.inMask.setText(file_name)
            gui.dockwidget.inMask.setEnabled(True)
        else:
            gui.dockwidget.checkInMask.setChecked(False)
            gui.dockwidget.inMask.setEnabled(False)
    elif sender == gui.dockwidget.checkInMask:
        gui.dockwidget.inMask.clear()
        gui.dockwidget.inMask.setEnabled(False)

    if sender == gui.dockwidget.checkOutMatrix and gui.dockwidget.checkOutMatrix.isChecked():
        file_name, _filter = QFileDialog.getSaveFileName(
            gui.dockwidget, "Save to a *.csv file", gui.lastSaveDir, "CSV (*.csv)"
        )
        gui.rememberLastSaveDir(file_name)
        if file_name != "":
            file_name, _file_extension = os.path.splitext(file_name)
            file_name = file_name + ".csv"
            gui.dockwidget.outMatrix.setText(file_name)
            gui.dockwidget.outMatrix.setEnabled(True)
            gui.dockwidget.inSplit.setEnabled(True)
            gui.dockwidget.inSplit.setValue(50)
        else:
            gui.dockwidget.checkOutMatrix.setChecked(False)
            gui.dockwidget.outMatrix.setEnabled(False)
            gui.dockwidget.outMatrix.setEnabled(False)
            gui.dockwidget.inSplit.setEnabled(False)
            gui.dockwidget.inSplit.setValue(100)
    elif sender == gui.dockwidget.checkOutMatrix:
        gui.dockwidget.outMatrix.clear()
        gui.dockwidget.checkOutMatrix.setChecked(False)
        gui.dockwidget.outMatrix.setEnabled(False)
        gui.dockwidget.outMatrix.setEnabled(False)
        gui.dockwidget.inSplit.setEnabled(False)
        gui.dockwidget.inSplit.setValue(100)

    if sender == gui.dockwidget.checkInConfidence and gui.dockwidget.checkInConfidence.isChecked():
        file_name, _filter = QFileDialog.getSaveFileName(
            gui.dockwidget,
            "Select output file (*.tif)",
            gui.lastSaveDir,
            "TIF (*.tif)",
        )
        gui.rememberLastSaveDir(file_name)
        if file_name != "":
            file_name, _file_extension = os.path.splitext(file_name)
            file_name = file_name + ".tif"
            gui.dockwidget.outConfidenceMap.setText(file_name)
            gui.dockwidget.outConfidenceMap.setEnabled(True)
        else:
            gui.dockwidget.checkInConfidence.setChecked(False)
            gui.dockwidget.outConfidenceMap.setEnabled(False)
    elif sender == gui.dockwidget.checkInConfidence:
        gui.dockwidget.outConfidenceMap.clear()
        gui.dockwidget.checkInConfidence.setChecked(False)
        gui.dockwidget.outConfidenceMap.setEnabled(False)
