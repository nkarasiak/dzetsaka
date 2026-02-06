"""Main panel training/classification execution flow."""

from __future__ import annotations

import os
import tempfile

from qgis.PyQt.QtWidgets import QMessageBox

try:
    from osgeo import gdal, ogr, osr
except ImportError:
    import gdal
    import ogr
    import osr

from dzetsaka import classifier_config
from dzetsaka.scripts.function_dataraster import get_layer_source_path


def run_magic(gui) -> None:
    """Perform training and classification from the main dock panel."""
    message = " "
    inRaster = ""
    inShape = ""
    inRasterOp = None
    model_path = gui.dockwidget.inModel.text().strip()

    inRasterLayer = gui.dockwidget.inRaster.currentLayer()
    if inRasterLayer is None:
        QMessageBox.warning(
            gui.iface.mainWindow(),
            "Missing Input",
            "Please select an input raster before running classification.",
            QMessageBox.StandardButton.Ok,
        )
        return

    try:
        inRaster = get_layer_source_path(inRasterLayer)
    except Exception:
        inRaster = ""
    if not inRaster:
        QMessageBox.warning(
            gui.iface.mainWindow(),
            "Invalid Raster",
            "Could not read the selected raster source. Please select a valid raster layer.",
            QMessageBox.StandardButton.Ok,
        )
        return

    if model_path == "":
        inShapeLayer = gui.dockwidget.inShape.currentLayer()
        if inShapeLayer is None:
            QMessageBox.warning(
                gui.iface.mainWindow(),
                "Missing Input",
                "If you don't use an existing model, please select training data (vector).",
                QMessageBox.StandardButton.Ok,
            )
            return
        try:
            inShape = get_layer_source_path(inShapeLayer)
        except Exception:
            inShape = ""
        if not inShape:
            QMessageBox.warning(
                gui.iface.mainWindow(),
                "Invalid Vector",
                "Could not read the selected training vector source. Please select a valid vector layer.",
                QMessageBox.StandardButton.Ok,
            )
            return

    try:
        inRasterOp = gdal.Open(inRaster)
        inRasterProj = osr.SpatialReference(inRasterOp.GetProjection()) if inRasterOp is not None else None

        if model_path == "":
            inShapeOp = ogr.Open(inShape)
            inShapeLyr = inShapeOp.GetLayer() if inShapeOp is not None else None
            inShapeProj = inShapeLyr.GetSpatialRef() if inShapeLyr is not None else None

            if inShapeProj is not None and inRasterProj is not None and inShapeProj.IsSameGeogCS(inRasterProj) == 0:
                message = message + "\n - Raster and ROI do not have the same projection."
    except Exception as exc:
        gui.log.error(f"Projection validation error: {exc}")
        if inShape:
            gui.log.error("inShape is : " + inShape)
        if inRaster:
            gui.log.error("inRaster is : " + inRaster)
        message = message + "\n - Can't compare projection between raster and vector."

    try:
        inMask = gui.dockwidget.inMask.text()

        if inMask == "":
            inMask = None
        autoMask = os.path.splitext(inRaster)
        autoMask = autoMask[0] + gui.maskSuffix + autoMask[1]

        if os.path.exists(autoMask):
            inMask = autoMask
            gui.log.info("Mask found : " + str(autoMask))

        if inMask is not None and inRasterOp is not None:
            mask = gdal.Open(inMask, gdal.GA_ReadOnly)
            if (inRasterOp.RasterXSize != mask.RasterXSize) or (inRasterOp.RasterYSize != mask.RasterYSize):
                message = message + "\n - Raster image and mask do not have the same size."

    except BaseException:
        message = message + "\n - Can't compare mask and raster size."

    if message != " ":
        reply = QMessageBox.question(
            gui.iface.mainWindow(),
            "Informations missing or invalid",
            message + "\n Would you like to continue anyway ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            message = " "

    if message == " ":
        gui.loadConfig()
        model = gui.dockwidget.inModel.text()

        inClassifier = classifier_config.get_classifier_code(gui.classifier)
        gui.log.info(f"Selected classifier: {gui.classifier} (code: {inClassifier})")

        if gui.dockwidget.outRaster.text() == "":
            tempFolder = tempfile.mkdtemp()
            outRaster = os.path.join(
                tempFolder,
                gui._default_output_name(inRaster, inClassifier),
            )
        else:
            outRaster = gui.dockwidget.outRaster.text()

        if gui.dockwidget.checkInConfidence.isChecked():
            confidenceMap = gui.dockwidget.outConfidenceMap.text()
        else:
            confidenceMap = None

        inClassifier = str(inClassifier)
        NODATA = -9999

        if model != "":
            model = gui.dockwidget.inModel.text()
            gui.log.info(f"Using existing model: {model}")
        else:
            if gui.dockwidget.outModel.text() == "":
                model = tempfile.mktemp("." + str(inClassifier))
            else:
                model = gui.dockwidget.outModel.text()
            gui.log.info("Training new model (no existing model loaded)")

        inField = gui.dockwidget.inField.currentText()
        inSeed = 0
        if gui.dockwidget.checkOutMatrix.isChecked():
            outMatrix = gui.dockwidget.outMatrix.text()
            inSplit = gui.dockwidget.inSplit.value()
        else:
            inSplit = 100
            outMatrix = None

        do_training = not gui.dockwidget.checkInModel.isChecked()
        if not gui._validate_classification_request(
            raster_path=inRaster,
            do_training=do_training,
            vector_path=inShape if do_training else None,
            class_field=inField if do_training else None,
            model_path=model if not do_training else None,
            source_label="Main Panel",
        ):
            return
        if not gui._ensure_classifier_runtime_ready(inClassifier, source_label="Main Panel", fallback_to_gmm=True):
            return
        gui.log.info(
            f"Starting {'training and ' if do_training else ''}classification with {inClassifier} classifier"
        )
        gui._start_classification_task(
            description=f"dzetsaka: {inClassifier} classification",
            do_training=do_training,
            raster_path=inRaster,
            vector_path=inShape if do_training else None,
            class_field=inField if do_training else None,
            model_path=model,
            split_config=inSplit,
            random_seed=inSeed,
            matrix_path=outMatrix,
            classifier=inClassifier,
            output_path=outRaster,
            mask_path=inMask,
            confidence_map=confidenceMap,
            nodata=NODATA,
            extra_params=None,
            error_context="Main panel classification workflow",
            success_prefix="Main",
        )
