A fork of Dan MacNish's excellent cartoon camera [project](https://danmacnish.com/2018/07/01/draw-this/).

Adds OpenCV for (semi-)realtime processing of webcam images.
Anonymizes with funny results.

What the camera sees [(source)](https://www.pexels.com/photo/group-of-people-watching-on-laptop-1595385/):

<img src="https://raw.githubusercontent.com/psaikko/cartoonify/master/img/test.jpg" alt="Input image" width="50%">

What the object detection model computes:

<img src="https://raw.githubusercontent.com/psaikko/cartoonify/master/img/annotated.png" alt="Object detection results" width="50%">

What the program outputs:

<img src="https://raw.githubusercontent.com/psaikko/cartoonify/master/img/out.jpg" alt="Cartoon output" width="50%">

To use:
```
virtualenv -p python3.6 venv
. ./venv/bin/activate
pip install -r requirements_desktop.txt
python run.py
```

For information on command-line options
```
python run.py --help
```
