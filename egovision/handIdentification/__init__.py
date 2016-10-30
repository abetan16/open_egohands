"""

Given a background/foreground hand-segmentation the next step is to decide
which contours belongs to the left and to the right hand. Recent works shows
that fusing the horizontal position of the hand and its angle with respect to
the lower border is possible to identify hand segments as left or right.
However, particular attention must be made when hand-to-hand occlusions appear.
The next figure shows some examples of a successful hand identification.

.. image:: ../_images/handIdentification/miniExDesamb2.png
    :align: center

In summary a identificationModel is to decide if a contour is left or right;
However its performance is strongly affected by the capability of the system to
detect and split hand-to-hand occlusions. This dependency is included in the
library by abstracting hand-to-hand occlusion, detection and split, as an
intermediate step between the hand-segmentation and the hand-identification.
The following diagram shows an example of the system.

.. image:: ../_images/handIdentification/diagram.png
    :align: center

"""

from kernelIdentificationModel import KernelIdentificationModel
from maxwellIdentificationModel import MaxwellIdentificationModel
from abstractIdentificationModel import AbstractIdentificationModel
from h2hOcclusion import SuperpixelsOcclusionDetector
