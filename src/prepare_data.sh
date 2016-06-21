#!/usr/bin/env sh
#create the peta lmdb inputs
# N.B. set the path to the peta train + val data dirs

EXAMPLE=../example
DATA=/data/peta
METADATA=../data
TOOLS="$CAFFE_ROOT/build/tools"

TRAIN_DATA_ROOT="$DATA/train/"
VAL_DATA_ROOT="$DATA/test/"

RESIZE=true
if $RESIZE; then
    RESIZE_HEIGHT=128
    RESIZE_WIDTH=128
else
    RESIZE_HEIGHT=0
    RESIZE_HEIGHT=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_peta.sh to the path" \
       "where the peta training data is stored."
  exit 1
fi


if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_peta.sh to the path" \
       "where the peta validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."
echo "$TRAIN_DATA_ROOT"
echo "$VAL_DATA_ROOT"
GLOG_logtostderr=1 $TOOLS/convert_imageset \
		--resize_height=$RESIZE_HEIGHT \
		--resize_width=$RESIZE_WIDTH \
		--shuffle \
		$TRAIN_DATA_ROOT \
		$METADATA/train.txt \
		$EXAMPLE/peta_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
		--resize_height=$RESIZE_HEIGHT \
		--resize_width=$RESIZE_WIDTH \
		--shuffle \
		$VAL_DATA_ROOT \
		$METADATA/test.txt \
		$EXAMPLE/peta_test_lmdb

echo "Done."
