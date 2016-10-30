
Working with videos
===================

EgoHands is hardly dependent on OpenCV. However, to improve the user experience
when working with videos or frames we provide facade classes. We recomend to
use directly our facade clases, to reduce the size of your code and keep
issolated the dependency with OpenCV.

The Video Object
----------------
.. autoclass:: egovision.Video
   :members:


The Frame object
----------------
.. autoclass:: egovision.Frame
   :members:

Visualizing a video
-------------------
.. autoclass:: egovision.output.VideoVisualizer
   :members:

Export a video
--------------
.. autoclass:: egovision.output.VideoWriter
   :members:


