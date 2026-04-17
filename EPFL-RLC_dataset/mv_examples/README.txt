The folder contains .json files with 2044 examples each.

____________________
positions_55x45.json
--------------------
Contains positions generated with the calibration files.
Used grid density:  55x45 (2475 positions in total)
Each position has its id 'numPosition', whose enumeration starts from 0.


_____________
positive.json
-------------
Annotations of ANNOTATED occupied positions.
Each example also contains the id ('numPosition') of its associated position of the file 'positions_55x45.json'.


_____________
negative.json
-------------
Annotations of ANNOTATED free positions positions (not-occupied).
Each example also contains the id ('numPosition') of its associated position of the file 'positions_55x45.json'.


Since a negative multi-view example could have pedestrian in some of the views, each example also contains for 
each view information in "pedestrian" wheather there is a pedestrin or not, denoted with 1 and 0 if there is
and if there is not a pedestrian, respectively.


_________________
autoHardNeg1.json
-----------------
AUTO-GENERATED negative examples.
Each example is generated from different positive examples. 
In other words, each rectangle within one multi-view example does contain a person, but it is not the same one in all of the three views.


_________________
autoHardNeg2.json
-----------------
AUTO-GENERATED  negative examples.
Each example is generated from one positive example, of which in either one or in two of the views, the rectangles have been shifted to one of the neighbor positions.
