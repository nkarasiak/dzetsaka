"""Unit tests for output raster default naming."""

from dzetsaka.qgis.output_naming import default_output_name


def test_default_output_name_uses_full_boosting_names() -> None:
    assert default_output_name("C:/tmp/map.tif", "XGB") == "map_XGBoost.tif"
    assert default_output_name("C:/tmp/map.tif", "LGB") == "map_LightGBM.tif"
    assert default_output_name("C:/tmp/map.tif", "CB") == "map_CatBoost.tif"


def test_default_output_name_keeps_other_classifier_codes() -> None:
    assert default_output_name("C:/tmp/map.tif", "RF") == "map_RF.tif"
