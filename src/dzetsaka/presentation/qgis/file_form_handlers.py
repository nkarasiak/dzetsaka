"""File chooser and checkbox handlers for the main dock form."""

from __future__ import annotations

import os

from qgis.PyQt.QtWidgets import QFileDialog


def select_output_file(gui) -> None:
    """Select output file and enforce expected file extension."""
    sender = gui.sender()
    dock = gui.dock_widget

    file_name, _filter = QFileDialog.getSaveFileName(
        dock, "Select output file", gui.lastSaveDir, "TIF (*.tif)"
    )
    gui.rememberLastSaveDir(file_name)

    if not file_name:
        return

    file_name, file_extension = os.path.splitext(file_name)

    if sender == dock.outRasterButton:
        if file_extension != ".tif":
            dock.outRaster.setText(file_name + ".tif")
        else:
            dock.outRaster.setText(file_name + file_extension)

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
    dock = gui.dock_widget

    if sender == dock.checkInModel and dock.checkInModel.isChecked():
        file_name, _filter = QFileDialog.getOpenFileName(dock, "Select your file", gui.lastSaveDir)
        gui.rememberLastSaveDir(file_name)
        if file_name != "":
            dock.inModel.setText(file_name)
            dock.inModel.setEnabled(True)
            dock.inShape.setEnabled(False)
            dock.inField.setEnabled(False)
        else:
            dock.checkInModel.setChecked(False)
            dock.inModel.setEnabled(False)
            dock.inShape.setEnabled(True)
            dock.inField.setEnabled(True)
    elif sender == dock.checkInModel:
        dock.inModel.clear()
        dock.inModel.setEnabled(False)
        dock.inShape.setEnabled(True)
        dock.inField.setEnabled(True)

    if sender == dock.checkOutModel and dock.checkOutModel.isChecked():
        file_name, _filter = QFileDialog.getSaveFileName(dock, "Select output file", gui.lastSaveDir)
        gui.rememberLastSaveDir(file_name)
        if file_name != "":
            dock.outModel.setText(file_name)
            dock.outModel.setEnabled(True)
        else:
            dock.checkOutModel.setChecked(False)
            dock.outModel.setEnabled(False)
    elif sender == dock.checkOutModel:
        dock.outModel.clear()
        dock.outModel.setEnabled(False)

    if sender == dock.checkInMask and dock.checkInMask.isChecked():
        file_name, _filter = QFileDialog.getOpenFileName(
            dock,
            "Select your mask raster",
            gui.lastSaveDir,
            "TIF (*.tif)",
        )
        gui.rememberLastSaveDir(file_name)
        if file_name != "":
            dock.maskPathEdit.setText(file_name)
            dock.maskPathEdit.setEnabled(True)
        else:
            dock.checkInMask.setChecked(False)
            dock.maskPathEdit.setEnabled(False)
    elif sender == dock.checkInMask:
        dock.maskPathEdit.clear()
        dock.maskPathEdit.setEnabled(False)

    if sender == dock.checkOutMatrix and dock.checkOutMatrix.isChecked():
        file_name, _filter = QFileDialog.getSaveFileName(
            dock, "Save to a *.csv file", gui.lastSaveDir, "CSV (*.csv)"
        )
        gui.rememberLastSaveDir(file_name)
        if file_name != "":
            file_name, _file_extension = os.path.splitext(file_name)
            file_name = file_name + ".csv"
            dock.confusionMatrixPathEdit.setText(file_name)
            dock.confusionMatrixPathEdit.setEnabled(True)
            dock.validationSplitPercentSpin.setEnabled(True)
            dock.validationSplitPercentSpin.setValue(50)
        else:
            dock.checkOutMatrix.setChecked(False)
            dock.confusionMatrixPathEdit.setEnabled(False)
            dock.confusionMatrixPathEdit.setEnabled(False)
            dock.validationSplitPercentSpin.setEnabled(False)
            dock.validationSplitPercentSpin.setValue(100)
    elif sender == dock.checkOutMatrix:
        dock.confusionMatrixPathEdit.clear()
        dock.checkOutMatrix.setChecked(False)
        dock.confusionMatrixPathEdit.setEnabled(False)
        dock.confusionMatrixPathEdit.setEnabled(False)
        dock.validationSplitPercentSpin.setEnabled(False)
        dock.validationSplitPercentSpin.setValue(100)

    if sender == dock.confidenceMapCheckBox and dock.confidenceMapCheckBox.isChecked():
        file_name, _filter = QFileDialog.getSaveFileName(
            dock,
            "Select output file (*.tif)",
            gui.lastSaveDir,
            "TIF (*.tif)",
        )
        gui.rememberLastSaveDir(file_name)
        if file_name != "":
            file_name, _file_extension = os.path.splitext(file_name)
            file_name = file_name + ".tif"
            dock.confidenceMapPathEdit.setText(file_name)
            dock.confidenceMapPathEdit.setEnabled(True)
        else:
            dock.confidenceMapCheckBox.setChecked(False)
            dock.confidenceMapPathEdit.setEnabled(False)
    elif sender == dock.confidenceMapCheckBox:
        dock.confidenceMapPathEdit.clear()
        dock.confidenceMapCheckBox.setChecked(False)
        dock.confidenceMapPathEdit.setEnabled(False)



