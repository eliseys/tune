# Tune 

Polar axis alignment tool

## Description

Tune is command-line polar alignment tool. It helps accurately align equatorial mount with the celestial pole by using plate-solving technology. Tune receives as input 2 photos of circumpolar area of the sky, or 3 photos during three-point polar alignment, taken by the camera on the mount with an arbitrary rotation between frames around the polar axis of the mount. Tune identifies stars on each image and calculate the exact position of the mount polar axis in relation to the celestial pole. Once the software has determined the exact position of the axis, it can then provide the user with information on how to adjust the telescope's position to achieve precise polar alignment. This information include the required adjustments to the telescope's altitude and azimuth. Accurate polar alignment is essential for successful long-exposure astrophotography and other astronomical observations.


### Dependencies

* (<samp>astrometry.net</samp>)[http://astrometry.net/]
* gPhoto2
* Sextractor
* Python

## Usage


```console
~ ./tune.py
```

```console
~ ./tune.py -n 3
```

```console
~ ./tune.py -f image1.jpg image2.jpg
```

```console
~ ./tune.py -f image1.jpg image2.jpg image3.jpg
```



