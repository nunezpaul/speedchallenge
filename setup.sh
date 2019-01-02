#!/usr/bin/env bash

# Download all videos and move them to data/videos
mkdir -p data/videos
for NAME in val test train; do
  if test ! -f data/videos/$NAME.mp4; then
    wget sunlight.caltech.edu/pnunez/speedchallenge/$NAME.mp4 && mv $NAME.mp4 data/videos
  else
    echo $NAME.mp4 already exists!
  fi
done

# Download labeled speeds for their respective videos
for NAME in val train; do
  if test ! -f data/$NAME.txt; then
    wget sunlight.caltech.edu/pnunez/speedchallenge/$NAME.txt
  else
    echo $NAME.txt already exists!
  fi
done

# Create the image, labeled_csv and tfrecords dirs for the respective files
for NAME in val test train; do
  mkdir -p data/images/$NAME
  mkdir -p data/labeled_csv/$NAME
  mkdir -p data/tfrecords/$NAME

# Convert the videos to jpegs using ffmpeg
ffmpeg -v || apt install ffmpeg
ffmpeg -i data/videos/train.mp4 -start_number 0 -qscale:v 2 data/images/train/img%d.jpg -hide_banner; echo train images done!
ffmpeg -i data/videos/val.mp4 -start_number 0 -qscale:v 2 data/images/val/img%d.jpg -hide_banner; echo val images done!
ffmpeg -i data/videos/test.mp4 -start_number 0 -qscale:v 2 data/images/test/img%d.jpg -hide_banner; echo test images done!

# Create all img label and place in respective directory
python create_data_pairs.py --speed_file data/train.txt \
--output_file data/labeled_csv/train/train_shard.csv --shard
python create_data_pairs.py --speed_file data/val.txt --output_file data/labeled_csv/val/val.csv
python create_test_img_pairs.py

# Writing the shards of the training tfrecords
for i in {0..9}; do
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