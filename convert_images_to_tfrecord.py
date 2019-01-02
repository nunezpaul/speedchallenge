# https://github.com/tensorflow/tensorflow/issues/9675
# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb

import argparse
import sys

import pandas as pd
import tensorflow as tf


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def convert_to_tfrecord(prev_img_paths, curr_img_paths, out_path, categories, speeds):
    # Args:
    # image_paths   List of file-paths for the train_images.
    # labels        Class-labels for the train_images.
    # out_path      File-path for the TFRecords output file.

    print("\nCreating: " + out_path)

    # Number of train_images. Used when printing the progress.
    num_images = len(prev_img_paths)

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the image-paths and class-labels.
        for i, (prev_img_path, curr_img_path, category, speed) in enumerate(zip(prev_img_paths,
                                                                                curr_img_paths,
                                                                                categories,
                                                                                speeds)):
            # Print the percentage-progress.
            print_progress(count=i, total=num_images - 1)

            prev_img_bytes = open(prev_img_path, 'rb').read()
            curr_img_bytes = open(curr_img_path, 'rb').read()
            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'prev_img': wrap_bytes(prev_img_bytes),
                    'curr_img': wrap_bytes(curr_img_bytes),
                }

            if category:
                data['category'] = wrap_int64(int(category))
                data['speed'] = wrap_float(float(speed))


            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert csv training file into tfrecord files.')
    parser.add_argument('--input_filename', type=str, default='data/labeled_csv/train/train_shard_0.csv',
                        help='file to be converted into a tfrecord')
    parser.add_argument('--output_filename', type=str, default='data/tfrecords/train_tfrecords/shard_0.tfrecord',
                        help='file location for tfrecord')
    args = parser.parse_args()

    file, out = args.input_filename, args.output_filename
    data = pd.read_csv(file)
    length = data.shape[0]
    width = data.shape[-1]

    convert_to_tfrecord(prev_img_paths=data.prev_img,
                        curr_img_paths=data.curr_img,
                        out_path=out,
                        speeds=data.speed if width > 2 else [None] * length,
                        categories=data.category if width > 2 else [None] * length)
