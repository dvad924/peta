#!/bin/sh
# python modelrun.py deploy.prototxt ../models/modelrev_iteration_iter_10000.caffemodel \
#       --list people_labels.txt --out people.res.txt

# python modelrun.py deploy.prototxt ../models/model_iteration_iter_10000.caffemodel \
#        --vid ../../drone/datasets/UCF/actions1.mpg --out actions1.res.txt

# python modelrun.py deploy.prototxt ../models/model_iteration_iter_10000.caffemodel \
#        --vid ../../drone/datasets/UCF/actions2.mpg --out actions2.res.txt

# python modelrun.py deploy.prototxt ../models/model_iteration_iter_10000.caffemodel \
#        --vid ../../drone/datasets/UCF/actions3.mpg --out actions3.res.txt

python modelrun.py deploy.prototxt ../models/modelrev_iteration_iter_10000.caffemodel \
       --list ../../data/actions1/labels.txt --out actions1crop.res.txt

python modelrun.py deploy.prototxt ../models/modelrev_iteration_iter_10000.caffemodel \
       --list ../../data/actions3/labels.txt --out actions3crop.res.txt
