#!/usr/bin/env bash

# Download all videos and move them to data/videos
mkdir -p data/videos
wget sunlight.caltech.edu/pnunez/speedchallenge/test.mp4 && mv test.mp4 data/videos
wget sunlight.caltech.edu/pnunez/speedchallenge/train.mp4 && mv train.mp4 data/videos
wget sunlight.caltech.edu/pnunez/speedchallenge/train.txt && mv train.txt data/videos

# Create the image directories that the respective frames from each video will go into
mkdir -p data/images/train
mkdir data/images/val
mkdir data/images/test

# Convert the videos to jpegs using ffmpeg
ffmpeg -v || apt install ffmpeg
ffmpeg -i data/videos/train.mp4 -start_number 0 -qscale:v 2 data/images/train/img%d.jpg -hide_banner; echo train images done!
ffmpeg -i data/videos/val.mp4 -start_number 0 -qscale:v 2 data/images/val/img%d.jpg -hide_banner; echo val images done!
ffmpeg -i data/videos/test.mp4 -start_number 0 -qscale:v 2 data/images/test/img%d.jpg -hide_banner; echo test images done!

# Creating labeled_csv directories
mkdir -p data/labeled_csv/val
mkdir data/labeled_csv/test
mkdir data/labeled_csv/train

# Create all img label and place in respective directory
python create_data_pairs.py --speed_file data/train.txt \
--output_file data/labeled_csv/train/train_shard.csv --shard
python create_data_pairs.py --speed_file data/val.txt --output_file data/labeled_csv/val/val.csv
python create_test_img_pairs.py

# Where the respective tfrecords will be stored
mkdir -p data/tfrecords/train
mkdir data/tfrecords/val
mkdir data/tfrecords/test

# Writing the shards of the training tfrecords
for i in {0..9}
do
INPUTFILE=data/labeled_csv/train/train_shard_$i.csv
OUTPUTFILE=data/tfrecords/train/shard_$i.tfrecord
python convert_images_to_tfrecord.py --input_filename $INPUTFILE --output_filename $OUTPUTFILE
done

# Writing the validation and test data to tfrecord
python convert_images_to_tfrecord.py --input_filename data/labeled_csv/val/val.csv \
--output data/tfrecords/val/val.tfrecord
python convert_images_to_tfrecord.py --input_filename data/labeled_csv/test/test.csv \
--output data/tfrecords/test/test.tfrecord

echo "Set up complete. Model is ready for training!"