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
    validation_message = " "
    raster_path = ""
    vector_path = ""
    raster_dataset = None
    dock = gui.dock_widget
    model_path = dock.inModel.text().strip()

    raster_layer = dock.inRaster.currentLayer()
    if raster_layer is None:
        QMessageBox.warning(
            gui.iface.mainWindow(),
            "Missing Input",
            "Please select an input raster before running classification.",
            QMessageBox.StandardButton.Ok,
        )
        return

    try:
        raster_path = get_layer_source_path(raster_layer)
    except Exception:
        raster_path = ""
    if not raster_path:
        QMessageBox.warning(
            gui.iface.mainWindow(),
            "Invalid Raster",
            "Could not read the selected raster source. Please select a valid raster layer.",
            QMessageBox.StandardButton.Ok,
        )
        return

    if model_path == "":
        vector_layer = dock.inShape.currentLayer()
        if vector_layer is None:
            QMessageBox.warning(
                gui.iface.mainWindow(),
                "Missing Input",
                "If you don't use an existing model, please select training data (vector).",
                QMessageBox.StandardButton.Ok,
            )
            return
        try:
            vector_path = get_layer_source_path(vector_layer)
        except Exception:
            vector_path = ""
        if not vector_path:
            QMessageBox.warning(
                gui.iface.mainWindow(),
                "Invalid Vector",
                "Could not read the selected training vector source. Please select a valid vector layer.",
                QMessageBox.StandardButton.Ok,
            )
            return

    try:
        raster_dataset = gdal.Open(raster_path)
        raster_projection = osr.SpatialReference(raster_dataset.GetProjection()) if raster_dataset is not None else None

        if model_path == "":
            vector_dataset = ogr.Open(vector_path)
            vector_layer = vector_dataset.GetLayer() if vector_dataset is not None else None
            vector_projection = vector_layer.GetSpatialRef() if vector_layer is not None else None

            if (
                vector_projection is not None
                and raster_projection is not None
                and vector_projection.IsSameGeogCS(raster_projection) == 0
            ):
                validation_message = validation_message + "\n - Raster and ROI do not have the same projection."
    except Exception as exc:
        gui.log.error(f"Projection validation error: {exc}")
        if vector_path:
            gui.log.error("vector_path is : " + vector_path)
        if raster_path:
            gui.log.error("raster_path is : " + raster_path)
        validation_message = validation_message + "\n - Can't compare projection between raster and vector."

    try:
        mask_path = dock.maskPathEdit.text()

        if mask_path == "":
            mask_path = None
        auto_mask_path = os.path.splitext(raster_path)
        auto_mask_path = auto_mask_path[0] + gui.maskSuffix + auto_mask_path[1]

        if os.path.exists(auto_mask_path):
            mask_path = auto_mask_path
            gui.log.info("Mask found : " + str(auto_mask_path))

        if mask_path is not None and raster_dataset is not None:
            mask_dataset = gdal.Open(mask_path, gdal.GA_ReadOnly)
            if (raster_dataset.RasterXSize != mask_dataset.RasterXSize) or (
                raster_dataset.RasterYSize != mask_dataset.RasterYSize
            ):
                validation_message = validation_message + "\n - Raster image and mask do not have the same size."

    except BaseException:
        mask_path = None
        validation_message = validation_message + "\n - Can't compare mask and raster size."

    if validation_message != " ":
        reply = QMessageBox.question(
            gui.iface.mainWindow(),
            "Informations missing or invalid",
            validation_message + "\n Would you like to continue anyway ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            validation_message = " "

    if validation_message == " ":
        gui.loadConfig()
        model_path = dock.inModel.text()

        classifier_code = classifier_config.get_classifier_code(gui.classifier)
        gui.log.info(f"Selected classifier: {gui.classifier} (code: {classifier_code})")

        if dock.outRaster.text() == "":
            temp_folder = tempfile.mkdtemp()
            output_raster_path = os.path.join(
                temp_folder,
                gui._default_output_name(raster_path, classifier_code),
            )
        else:
            output_raster_path = dock.outRaster.text()

        confidence_map_path = dock.confidenceMapPathEdit.text() if dock.confidenceMapCheckBox.isChecked() else None

        classifier_code = str(classifier_code)
        nodata_value = -9999

        if model_path != "":
            model_path = dock.inModel.text()
            gui.log.info(f"Using existing model: {model_path}")
        else:
            if dock.outModel.text() == "":
                model_path = tempfile.mktemp("." + str(classifier_code))
            else:
                model_path = dock.outModel.text()
            gui.log.info("Training new model (no existing model loaded)")

        class_field = dock.inField.currentText()
        random_seed = 0
        if dock.checkOutMatrix.isChecked():
            matrix_path = dock.confusionMatrixPathEdit.text()
            split_percent = dock.validationSplitPercentSpin.value()
        else:
            split_percent = 100
            matrix_path = None

        do_training = not dock.checkInModel.isChecked()
        if not gui._validate_classification_request(
            raster_path=raster_path,
            do_training=do_training,
            vector_path=vector_path if do_training else None,
            class_field=class_field if do_training else None,
            model_path=model_path if not do_training else None,
            source_label="Main Panel",
        ):
            return
        if not gui._ensure_classifier_runtime_ready(classifier_code, source_label="Main Panel", fallback_to_gmm=True):
            return
        gui.log.info(
            f"Starting {'training and ' if do_training else ''}classification with {classifier_code} classifier"
        )
        gui._start_classification_task(
            description=f"dzetsaka: {classifier_code} classification",
            do_training=do_training,
            raster_path=raster_path,
            vector_path=vector_path if do_training else None,
            class_field=class_field if do_training else None,
            model_path=model_path,
            split_config=split_percent,
            random_seed=random_seed,
            matrix_path=matrix_path,
            classifier=classifier_code,
            output_path=output_raster_path,
            mask_path=mask_path,
            confidence_map=confidence_map_path,
            nodata=nodata_value,
            extra_params=None,
            error_context="Main panel classification workflow",
            success_prefix="Main",
        )



