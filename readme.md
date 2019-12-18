# dzetsaka : classification tool
[![DOI](https://zenodo.org/badge/59029116.svg)](https://zenodo.org/badge/latestdoi/59029116)

![Inselberg in Guiana Amazonian Park](https://cdn.rawgit.com/lennepkade/dzetsaka/master/img/guyane.jpg)

dzetsaka <img src="https://cdn.rawgit.com/lennepkade/dzetsaka/master/img/icon.png" alt="dzetsaka logo" width="30px"/> is very fast and easy to use but also a **powerful classification plugin for Qgis**. Initially based on Gaussian Mixture Model classifier developped by  [Mathieu Fauvel](http://fauvel.mathieu.free.fr) (now supports Random Forest, KNN and SVM), this plugin is a more generalist tool than [Historical Map](https://github.com/lennepkade/HistoricalMap) which was dedicated to classify forests from old maps.
This plugin has by developped by [Nicolas Karasiak](http://www.karasiak.net/).

A **quick tutorial is available online** ([dzetsaka : how to make your first classification in qgis ?](http://www.karasiak.net/dzetsaka-how-to-make-your-first-classification-in-qgis/)), or you can just [download samples](https://github.com/lennepkade/dzetsaka/archive/docs.zip) to test the plugin on your own.

## What does dzetsaka mean ?
As this tool was developped during my work in the Guiana Amazonian Park to classify different kind of vegetation, I gave an Teko name (a native-american language from a nation which lives in french Guiana) which represent the objects we use to see the world through, such as satellites, microscope, camera... 

## Discover dzetsaka
`dzetsaka : Classification tool` runs with scipy library. You can download package like [Spider by Anaconda](https://docs.continuum.io/anaconda/) for a very easy setup. 

Then, as this plugin is very simple, you will just need two things for making a good classification : 
- A **raster**
- A **shapefile** which contains your **ROI** (Region Of Interest)

The shapefile must have a column which contains your classification numbers *(1,3,4...)*. Otherwise if you use text or anything else it certainly won't work.

## Installation of scikit-learn
On **Linux** simply open terminal and type : 
`python3 -m pip install scikit-learn --user`

### On Windows

**For Qgis 3**: 
Open OsGeo shell, then :

`py3_env.bat`

`python3 -m pip install scikit-learn --user`

Thanks to Alexander Bruy for the tip.

**For Qgis 2**:
In the OsGeo setup, search for PIP and install it. Then you have few more steps to do. In the explorer, search for OsGeo4W Shell, right click to open it as an administrator. Now use pip in OsGeo Shell like on Linux. Just type :<br/>
`pip install scikit-learn`

If you do not have pip installed, open osgeo4w-setup-x86_64.exe, select Advanced install and install *pip*.


You can now use **Random Forest**, **SVM**, or **KNN** !

## Tips

- If your raster is *spot6scene.tif*, you can create your mask under the name *spot6scene_mask.tif* and the script will detect it automatically.
- If you want to keep your spectral ROI model from an image, you can save your model to use it on another image.

Online dev documentation is available throught the [doxygen branch](https://rawgit.com/lennepkade/dzetsaka/doxygen/index.html).

## Like us, use us ? Cite us !

If you use dzetsaka in your research and find it useful, please cite Dzetsaka using the following bibtex reference:

```
@misc{karasiak2016dzetsaka,
title={Dzetsaka Qgis Classification plugin},
author={Karasiak, Nicolas},
url={https://github.com/nkarasiak/dzetsaka},
year={2016},
doi={10.5281/zenodo.2552284}
}
```

### Thanks to...
I would like to thank the [Guiana Amazonian Park](http://www.parc-amazonien-guyane.fr/) for their trust in my work, and the Master 2 Geomatics [Sigma](http://sigma.univ-toulouse.fr/en/welcome.html) for their excellent lessons in geomatics.

![Sponsors of Qgis](https://cdn.rawgit.com/lennepkade/dzetsaka/master/img/logo.png)
