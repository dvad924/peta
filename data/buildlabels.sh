#!/bin/sh

python labels.py --dir /data/peta/cars/car_data neglabels.txt 0
python labels.py --dir /data/peta/bikes/bike_data neglabels.txt 0

python labels.py --dir /data/peta/peta_data_test poslabels.txt 1
python labels.py --dir /data/peta/peta_data_train poslabels.txt 1

python mapmaker.py neglabels.txt poslabels.txt
