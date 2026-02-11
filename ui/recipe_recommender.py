"""Smart recommendation engine for suggesting recipes based on raster characteristics.

This module analyzes input raster files and recommends appropriate classification
recipes based on detected characteristics like band count, file size, sensor type,
and filename hints.

Author:
    Nicolas Karasiak
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from osgeo import gdal
except ImportError:
    import gdal  # type: ignore


class RasterAnalyzer:
    """Analyzes raster characteristics for recipe recommendation."""

    def __init__(self):
        """Initialize the analyzer."""
        # Sensor detection patterns
        self.sensor_patterns = {
            "sentinel2": [
                r"sentinel[\-_]?2",
                r"s2[a-z]?[\-_]",
                r"_t\d{2}[a-z]{3}_",  # Tile naming pattern
            ],
            "landsat8": [
                r"landsat[\-_]?8",
                r"lc08",
                r"lo8",
                r"_b\d{1,2}\.tif",  # Band naming
            ],
            "landsat9": [
                r"landsat[\-_]?9",
                r"lc09",
                r"lo9",
            ],
            "planet": [
                r"planet",
                r"_analytic",
            ],
            "modis": [
                r"mod\d{2}[a-z]\d",
                r"myd\d{2}[a-z]\d",
            ],
            "spot": [
                r"spot[\-_]?\d",
            ],
        }

        # Land cover type patterns
        self.landcover_patterns = {
            "agriculture": [r"crop", r"agri", r"farm", r"field"],
            "forest": [r"forest", r"tree", r"wood", r"canopy"],
            "urban": [r"urban", r"city", r"built", r"settlement"],
            "water": [r"water", r"lake", r"river", r"ocean", r"coast"],
            "wetland": [r"wetland", r"marsh", r"swamp"],
        }

    def analyze_raster(self, raster_path: str) -> Dict[str, Any]:
        """Analyze raster file and extract characteristics.

        Parameters
        ----------
        raster_path : str
            Path to the raster file to analyze

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - band_count: Number of bands in the raster
            - file_size_mb: File size in megabytes
            - filename_hints: List of detected keywords from filename
            - resolution_m: Pixel resolution in meters (if detectable)
            - crs: Coordinate reference system
            - detected_sensor: Detected sensor type
            - landcover_type: Detected land cover type hint
            - width: Raster width in pixels
            - height: Raster height in pixels
            - error: Error message if analysis failed

        Examples
        --------
        >>> analyzer = RasterAnalyzer()
        >>> info = analyzer.analyze_raster("/path/to/sentinel2_crop.tif")
        >>> info["detected_sensor"]
        'sentinel2'
        >>> info["landcover_type"]
        'agriculture'

        """
        result = {
            "band_count": 0,
            "file_size_mb": 0.0,
            "filename_hints": [],
            "resolution_m": None,
            "crs": "",
            "detected_sensor": "unknown",
            "landcover_type": "unknown",
            "width": 0,
            "height": 0,
            "error": None,
        }

        # Check if file exists
        if not os.path.exists(raster_path):
            result["error"] = "File does not exist"
            return result

        # Get file size
        try:
            file_size_bytes = os.path.getsize(raster_path)
            result["file_size_mb"] = file_size_bytes / (1024 * 1024)
        except Exception as e:
            result["error"] = f"Could not get file size: {str(e)}"

        # Analyze filename
        filename = os.path.basename(raster_path).lower()
        result["filename_hints"] = self._extract_filename_hints(filename)
        result["detected_sensor"] = self._detect_sensor(filename)
        result["landcover_type"] = self._detect_landcover_type(filename)

        # Open with GDAL
        try:
            dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
            if dataset is None:
                result["error"] = "Could not open raster with GDAL"
                return result

            # Get basic info
            result["band_count"] = dataset.RasterCount
            result["width"] = dataset.RasterXSize
            result["height"] = dataset.RasterYSize

            # Get CRS
            projection = dataset.GetProjection()
            if projection:
                result["crs"] = projection[:100]  # Truncate for display

            # Try to get resolution
            geotransform = dataset.GetGeoTransform()
            if geotransform:
                pixel_width = abs(geotransform[1])
                pixel_height = abs(geotransform[5])
                result["resolution_m"] = (pixel_width + pixel_height) / 2

            dataset = None  # Close dataset

        except Exception as e:
            result["error"] = f"GDAL error: {str(e)}"

        return result

    def _extract_filename_hints(self, filename: str) -> List[str]:
        """Extract useful hints from filename.

        Parameters
        ----------
        filename : str
            Lowercase filename to analyze

        Returns
        -------
        List[str]
            List of detected hints

        """
        hints = []

        # Check for common keywords
        keywords = [
            "sentinel",
            "landsat",
            "planet",
            "modis",
            "spot",
            "crop",
            "agriculture",
            "forest",
            "urban",
            "water",
            "rgb",
            "multispectral",
            "hyperspectral",
            "ndvi",
            "classified",
            "ortho",
        ]

        for keyword in keywords:
            if keyword in filename:
                hints.append(keyword)

        return hints

    def _detect_sensor(self, filename: str) -> str:
        """Detect sensor type from filename.

        Parameters
        ----------
        filename : str
            Lowercase filename to analyze

        Returns
        -------
        str
            Detected sensor type or 'unknown'

        """
        for sensor, patterns in self.sensor_patterns.items():
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    return sensor

        return "unknown"

    def _detect_landcover_type(self, filename: str) -> str:
        """Detect land cover type hint from filename.

        Parameters
        ----------
        filename : str
            Lowercase filename to analyze

        Returns
        -------
        str
            Detected land cover type or 'unknown'

        """
        for landcover, patterns in self.landcover_patterns.items():
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    return landcover

        return "unknown"


class RecipeRecommender:
    """Recommends classification recipes based on raster characteristics."""

    def __init__(self):
        """Initialize the recommender."""
        self.analyzer = RasterAnalyzer()

    def recommend(
        self, raster_info: Dict[str, Any], available_recipes: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """Recommend recipes based on raster characteristics.

        Parameters
        ----------
        raster_info : Dict[str, Any]
            Raster information from RasterAnalyzer.analyze_raster()
        available_recipes : List[Dict[str, Any]]
            List of available recipe dictionaries

        Returns
        -------
        List[Tuple[Dict[str, Any], float, str]]
            List of (recipe, confidence_score, reason) tuples sorted by confidence

        Examples
        --------
        >>> recommender = RecipeRecommender()
        >>> info = {"band_count": 12, "detected_sensor": "sentinel2"}
        >>> recommendations = recommender.recommend(info, recipes)
        >>> recipe, score, reason = recommendations[0]
        >>> score > 80.0
        True

        """
        if not available_recipes:
            return []

        recommendations = []

        for recipe in available_recipes:
            score, reasons = self._score_recipe(raster_info, recipe)
            if score > 0:
                reason_text = " • ".join(reasons) if reasons else "General compatibility"
                recommendations.append((recipe, score, reason_text))

        # Sort by confidence score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations

    def _score_recipe(self, raster_info: Dict[str, Any], recipe: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score a recipe's suitability for the given raster.

        Parameters
        ----------
        raster_info : Dict[str, Any]
            Raster characteristics
        recipe : Dict[str, Any]
            Recipe to score

        Returns
        -------
        Tuple[float, List[str]]
            (confidence_score, list_of_reasons)

        """
        score = 0.0
        reasons = []

        band_count = raster_info.get("band_count", 0)
        file_size_mb = raster_info.get("file_size_mb", 0.0)
        detected_sensor = raster_info.get("detected_sensor", "unknown")
        landcover_type = raster_info.get("landcover_type", "unknown")
        filename_hints = raster_info.get("filename_hints", [])

        recipe_name = recipe.get("name", "").lower()
        recipe_desc = recipe.get("description", "").lower()
        recipe_classifier = recipe.get("classifier", "RF")
        recipe_extra = recipe.get("extraParam", {})

        # Base score for all recipes
        score = 30.0

        # Sentinel-2 detection (10-13 bands is typical for Sentinel-2)
        if detected_sensor == "sentinel2" or "sentinel" in filename_hints:
            if 10 <= band_count <= 13:
                score += 40.0
                reasons.append("Perfect Sentinel-2 match (12-13 bands)")
            elif 4 <= band_count <= 10:
                score += 25.0
                reasons.append("Possible Sentinel-2 subset")

            # Boost recipes that mention Sentinel in description
            if "sentinel" in recipe_name or "sentinel" in recipe_desc:
                score += 20.0
                reasons.append("Recipe designed for Sentinel data")

        # Landsat detection (7-8 bands typical)
        elif detected_sensor in ["landsat8", "landsat9"] or "landsat" in filename_hints:
            if 7 <= band_count <= 11:
                score += 40.0
                reasons.append("Perfect Landsat match (7-11 bands)")
            elif 4 <= band_count <= 7:
                score += 25.0
                reasons.append("Possible Landsat subset")

            if "landsat" in recipe_name or "landsat" in recipe_desc:
                score += 20.0
                reasons.append("Recipe designed for Landsat data")

        # Hyperspectral detection (>20 bands)
        elif band_count > 20:
            score += 30.0
            reasons.append(f"Hyperspectral imagery ({band_count} bands)")

            # Boost feature selection recipes for hyperspectral
            if "feature" in recipe_name or "selection" in recipe_name:
                score += 15.0
                reasons.append("Feature selection helps with many bands")

        # RGB/4-band imagery
        elif band_count in [3, 4]:
            score += 20.0
            reasons.append("RGB or 4-band multispectral imagery")

        # File size considerations
        if file_size_mb > 1000:  # Large file (>1 GB)
            # Recommend fast algorithms
            fast_algos = ["RF", "ET", "LGB", "XGB"]
            if recipe_classifier in fast_algos:
                score += 15.0
                reasons.append(f"Fast algorithm suitable for large files ({file_size_mb:.0f} MB)")

            # Penalize slow algorithms
            slow_algos = ["SVM", "MLP"]
            if recipe_classifier in slow_algos:
                score -= 20.0
                reasons.append(f"May be slow on large files ({file_size_mb:.0f} MB)")

        elif file_size_mb > 500:  # Medium-large file
            # Slight preference for fast algorithms
            fast_algos = ["RF", "ET", "LGB", "XGB"]
            if recipe_classifier in fast_algos:
                score += 8.0
                reasons.append("Efficient for medium-large files")

        # Land cover specific recommendations
        if landcover_type != "unknown":
            landcover_keywords = {
                "agriculture": ["crop", "agri", "farm"],
                "forest": ["forest", "tree", "vegetation"],
                "urban": ["urban", "city", "built"],
                "water": ["water", "aquatic"],
            }

            keywords = landcover_keywords.get(landcover_type, [])
            recipe_text = recipe_name + " " + recipe_desc

            for keyword in keywords:
                if keyword in recipe_text:
                    score += 15.0
                    reasons.append(f"Optimized for {landcover_type} classification")
                    break

        # Imbalanced data handling
        if recipe_extra.get("SMOTE") or "imbalanced" in recipe_name:
            # Generic boost for imbalanced handling
            score += 5.0
            reasons.append("Handles imbalanced classes")

        # Accuracy vs speed tradeoffs
        if "fast" in recipe_name or "quick" in recipe_name:
            score += 5.0
            reasons.append("Optimized for speed")

        if "accuracy" in recipe_name or "best" in recipe_name:
            score += 8.0
            reasons.append("Optimized for accuracy")

        # Explainability features
        if recipe_extra.get("GENERATE_SHAP_REPORT") or "explain" in recipe_name:
            score += 5.0
            reasons.append("Includes model explainability (SHAP)")

        # Cap score at 100
        score = min(score, 100.0)

        return score, reasons

    def get_confidence_class(self, score: float) -> str:
        """Get confidence class from score.

        Parameters
        ----------
        score : float
            Confidence score (0-100)

        Returns
        -------
        str
            Confidence class description

        """
        if score >= 95:
            return "Excellent match"
        elif score >= 80:
            return "Good match"
        elif score >= 60:
            return "Possible match"
        elif score >= 40:
            return "Low confidence"
        else:
            return "Not recommended"

    def get_star_rating(self, score: float) -> str:
        """Get star rating from score.

        Parameters
        ----------
        score : float
            Confidence score (0-100)

        Returns
        -------
        str
            Star rating string

        """
        stars = int(round(score / 20.0))  # Convert to 0-5 scale
        return "⭐" * max(1, min(5, stars))
