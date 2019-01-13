#!/usr/bin/env bash

# Default settings for data processing. Needs to be consistent from start to end.
BUCKET_SIZE=3
MODEL_PARAMS_DIR=model_params/

while getopts "b:m:" OPTION; do
    case $OPTION in
    b)
        BUCKET_SIZE="$OPTARG"
        ;;
    m)
        MODEL_PARAMS_DIR="$OPTARG"
        ;;
    *)
        echo "Incorrect options provided"
        exit 1
        ;;
    esac
done

echo "Processing data with a speed bucket size of $BUCKET_SIZE. Use -b to change the value."

mkdir $MODEL_PARAMS_DIR
echo $BUCKET_SIZE > $MODEL_PARAMS_DIR\bucket_size.txt

# Download all videos and move them to data/videos
mkdir -p data/videos
for NAME in test train; do
  if test ! -f data/videos/$NAME.mp4; then
    wget sunlight.caltech.edu/pnunez/speedchallenge/$NAME.mp4 && mv $NAME.mp4 data/videos
  else
    echo $NAME.mp4 already exists!
  fi
done

# Download labeled speeds for their respective videos
for NAME in train; do
  if test ! -f data/$NAME.txt; then
    wget sunlight.caltech.edu/pnunez/speedchallenge/$NAME.txt && mv $NAME.txt data
  else
    echo $NAME.txt already exists!
  fi
done

# Create the image, labeled_csv and tfrecords dirs for the respective files
for NAME in test train val; do
  mkdir -p data/images/$NAME
  mkdir -p data/labeled_csv/$NAME
  mkdir -p data/tfrecords/$NAME
done

# Convert the videos to jpegs using ffmpeg
ffmpeg -v || apt install ffmpeg
ffmpeg -i data/videos/train.mp4 -start_number 0 -qscale:v 2 data/images/train/img%d.jpg -hide_banner; echo train images done!
ffmpeg -i data/videos/test.mp4 -start_number 0 -qscale:v 2 data/images/test/img%d.jpg -hide_banner; echo test images done!

# Create all img label and place in respective directory
python create_data_pairs.py --speed_file data/train.txt --output_file data/labeled_csv/train/train.csv \
--shuffle --data_split --split_inc 50 --lookback 5 --write_class_weights --bucket_size $BUCKET_SIZE \
--model_params_dir $MODEL_PARAMS_DIR
python create_data_pairs.py --speed_file data/train.txt --output_file data/labeled_csv/val/sorted_train.csv \
--bucket_size $BUCKET_SIZE
python create_data_pairs.py --img_dir data/images/test/ --output_file data/labeled_csv/test/test.csv --unlabeled

# Writing the train, test, val and sorted val tfrecords
python convert_images_to_tfrecord.py --input_filename data/labeled_csv/train/train.csv \
--output_filename data/tfrecords/train/train.tfrecord
python convert_images_to_tfrecord.py --input_filename data/labeled_csv/val/val.csv \
--output data/tfrecords/val/val.tfrecord
python convert_images_to_tfrecord.py --input_filename data/labeled_csv/val/sorted_train.csv \
--output data/tfrecords/val/sorted_train.tfrecord
python convert_images_to_tfrecord.py --input_filename data/labeled_csv/test/test.csv \
--output data/tfrecords/test/test.tfrecord

echo 'Set up complete. Model is ready for training!'
