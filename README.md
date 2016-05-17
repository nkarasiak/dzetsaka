## dzetsaka, classification tool
![Inselberg in Guiana Amazonian Park](https://cdn.rawgit.com/lennepkade/dzetsaka/master/img/guyane.jpg)

dzetsaka <img src="https://cdn.rawgit.com/lennepkade/dzetsaka/master/img/icon.png" alt="dzetsaka logo" width="30px"/> is very fast and easy to use but also a **powerful classification plugin for Qgis**. Based on Gaussian Mixture Model classifier developped by  [Mathieu Fauvel](http://fauvel.mathieu.free.fr), this plugin is a more generalist tool than [Historical Map](https://github.com/lennepkade/HistoricalMap) which was dedicated to classify forests from old maps.
This plugin has by developped by [Nicolaï Van Lennepkade](http://www.lennepka.de/) (aka Nicolas Karasiak).

### What does dzetsaka mean ?
As this tool was developped during my work in the Guiana Amazonian Park to classify different kind of vegetation, I gave an Teko name (a native-american language from a nation which lives in french Guiana) which means.

### Discover the magic of dzetsaka
`dzetsaka : Classification tool` runs with scipy library. You can download package like [Spider by Anaconda](https://docs.continuum.io/anaconda/) for a very easy setup. 

Then, as this plugin is very simple, you will just need two things for making a good classification : 
- A **raster**
- A **shapefile** which contains your **ROI** (Region Of Interest)

The shapefile must have a column which contains your classification numbers *(1,3,4...)*. Otherwise if you use text or anything else it certainly won't work.

### Tips

- If your raster is *spot6scene.tif*, you can create your mask under the name *spot6scene_mask.tif* and the script will detect it automatically.
- If you want to keep your spectral ROI model from an image, you can save your model to use it on another image.

### Todo
- Generate confusion matrix on demand
- Implement different but fast classifiers. Badly those in scipy (like Random Forest) took hours to classify.

