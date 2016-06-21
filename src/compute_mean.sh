#!/bin/sh

# Compute the mean image from the peta training lmdb

EXAMPLE=../example
DATA=../data
TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/compute_image_mean $EXAMPLE/peta_train_lmdb \
			  $DATA/peta_mean.binaryproto

echo "Done."
