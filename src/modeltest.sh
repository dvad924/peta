#!/bin/sh
python modelrun.py deploy.prototxt ../models/model_iteration_iter_10000.caffemodel \
      --list people_labels.txt --out inria_people.res.txt

python modelrun.py deploy.prototxt ../models/model_iteration_iter_10000.caffemodel \
       --vid ../../drone/datasets/UCF/actions1.mpg --out actions1.res.txt
