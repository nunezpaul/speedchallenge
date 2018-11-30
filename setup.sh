#!/usr/bin/env bash

mkdir data
wget sunlight.caltech.edu/pnunez/speedchallenge/test.mp4 && mv test.mp4 data
wget sunlight.caltech.edu/pnunez/speedchallenge/train.mp4 && mv train.mp4 data
wget sunlight.caltech.edu/pnunez/speedchallenge/train.txt && mv train.txt data

mkdir data/train_images
mkdir data/test_images

ffmpeg -v || apt install ffmpeg
ffmpeg -i data/train.mp4 -start_number 0 -qscale:v 2 data/train_images/img%d.jpg -hide_banner && ls data/train_images
ffmpeg -i data/test.mp4 -start_number 0 -qscale:v 2 data/test_images/img%d.jpg -hide_banner && ls data/test_images

mkdir data/sharded_image_value_pairs
python create_data_pairs.py

mkdir data/train_tfrecords
mkdir data/val_tfrecords