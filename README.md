# webgl_volume_demo

Demo application of volume rendering using three.js 

Derived from [`three.js` volume ray casting example](https://threejs.org/examples/?q=texture#webgl_texture3d). The following changes were added:

 * Bundle files togther to allow it to be used as a static web site
 * Add new colour map (and creator script)
 * Add new volume data (and creator script)
 * Add button and refactor code to load other volume data files

To use, serve the files using a simple web server. For example:
```
$ python -m http.server 8080
```
and open your browser at http://localhost:8080

# Utilities

There are Python scripts to create Nearly Raw Raster Data file format ([NRRD](https://teem.sourceforge.net/nrrd/)) and colour maps. These rely on pynrrd, h5py, numpy, Pillow and cmap.

