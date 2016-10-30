.. egovision documentation master file, created by
   sphinx-quickstart on Fri Jan 30 11:09:30 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to egohands python library!
====================================

Egohands is a python library to understand the movement of the hands when recorded by a wearable camera (usually head or chest mounted). This video perspective is commonly known as First Person Video (FPV) or just as egovision :cite:`Betancourt2014`. This video shows a nice example of what can be done using egohands and a proper training.

.. youtube:: https://www.youtube.com/watch?v=jmkayWZA5A0

This libary contains several algorithms and procedures developed for the particular problem of understanding how the user of a wearable camera is performing a manual task. This library is developed under the unified framework proposed by :cite:`Betancourt2015b`. 

.. image:: _images/hand_detection/diagram.png
    :align: center

EgoHands is designed under an Object Oriented structure and is divided according to levels of the framework proposed by Betancourt :cite:`Betancourt2015a`:cite:`Betancourt2015b`:cite:`Betancourt2015`:cite:`Betancourt2014a`. Here you will find methods for hand-detection, hand-segmentation, hand-identification, hand-tracking as well as some basic functionalities as feature extraction. 


Egohands is highly dependent of opencv and numpy and other machine learning libraries. Please verify the
"dependecies" section for more information.

Things to do with egovision
---------------------------
.. toctree::
   :maxdepth: 2
    
   Using videos with egovision <_video/videoModule>
   Extracting some features <_features/features>
   Hand-based methods <_handBased/handBased>
   Developer tools <_developers/developers>
   Egovision Team <_team/team>

Bibliography related with the libary
------------------------------------

.. bibliography:: zbiblio.bib


