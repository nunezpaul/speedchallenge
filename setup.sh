#!/usr/bin/env bash

# Download all videos and move them to data/videos
mkdir -p data/videos
wget sunlight.caltech.edu/pnunez/speedchallenge/test.mp4 && mv test.mp4 data/videos
wget sunlight.caltech.edu/pnunez/speedchallenge/train.mp4 && mv train.mp4 data/videos
wget sunlight.caltech.edu/pnunez/speedchallenge/train.txt && mv train.txt data/videos

# Create the image directories that the respective frames from each video will go into
mkdir -p data/images/train_images
mkdir data/images/valid_images
mkdir data/images/test_images

# Convert the videos to jpegs using ffmpeg
ffmpeg -v || apt install ffmpeg
ffmpeg -i data/videos/train.mp4 -start_number 0 -qscale:v 2 data/images/train/img%d.jpg -hide_banner; echo train images done!
ffmpeg -i data/videos/valid.mp4 -start_number 0 -qscale:v 2 data/images/valid/img%d.jpg -hide_banner; echo valid images done!
ffmpeg -i data/videos/test.mp4 -start_number 0 -qscale:v 2 data/images/test/img%d.jpg -hide_banner; echo test images done!

# Creating images label pairs
mkdir data/train_image_value_pairs
mkdir data/valid_image_value_pairs
python create_data_pairs.py --speed_file data/train.txt \
--output_file data/labeled_csv/train/train_shard.csv --data_split
python create_data_pairs.py --speed_file data/valid.txt --output_file data/labeled_csv/val/val_1.csv
python create_test_img_pairs.py --test_img_dir hold

# Where the respective tfrecords will be stored
mkdir data/train_tfrecords
mkdir data/val_tfrecords
mkdir data/test_tfrecords

# Writing the shards of the training tfrecords
for i in {0..9}
do
INPUTFILE=data/sharded_image_value_pairs/image_value_pairs_$i.csv
OUTPUTFILE=data/train_tfrecords/image_value_pairs_$i.tfrecord
python convert_images_to_tfrecord.py --input_filename $INPUTFILE --output_filename $OUTPUTFILE
done

# Writing the testing data to tfrecord
python convert_images_to_tfrecord.py --input_filename data/test_image_value_pairs.csv \
--output data/test_tfrecords/test_image_value_pairs.tfrecord