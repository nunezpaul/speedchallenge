import argparse

import keras as k
import tensorflow as tf

def continue_training(model):
    # Let's learn!
    model.fit(epochs=1, steps_per_epoch=8096)
    model.save('4_img_skip_model_1.h5')
    return model


def load_validation_data():
    data_path = 'data/val_tfrecords/val_image_value_pairs.tfrecord'
    dataset = tf.data.TFRecordDataset([data_path])
    dataset = dataset.map(_parse_val_function)
    dataset = dataset.repeat().batch(16)
    val_iter = dataset.make_one_shot_iterator()

    img, label = val_iter.get_next()
    img = tf.reshape(img, (-1, 200, 200, 6))

    return img, label


def _parse_val_function(example_proto):
    # Dictionary of features.
    feature = {
        'prev_img': tf.FixedLenFeature([], tf.string),
        'curr_img': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.float32)
    }

    # Parse the input tf.Example proto using the dictionary above.
    output = tf.parse_single_example(example_proto, feature)

    # Randomly crop the images within a limit
    crop_x = 200
    crop_y = 150
    crop_width = 200
    crop_height = 200
    crop_window = [crop_y, crop_x, crop_height, crop_width]

    prev_img = tf.image.decode_and_crop_jpeg(output['prev_img'], crop_window)
    curr_img = tf.image.decode_and_crop_jpeg(output['curr_img'], crop_window)

    # Normalize the data
    prev_img = (tf.to_float(prev_img) - 225 / 2) / 255
    curr_img = (tf.to_float(curr_img) - 225 / 2) / 255

    # Combine the two images
    concated_img = tf.concat([prev_img, curr_img], 2)

    return concated_img, output['label']

if __name__ == '__main__':
    model = k.models.load_model('4_img_skip_model.h5')
    val_x, val_y = load_validation_data()
    loss, acc = model.evaluate(val_x, val_y)
    print(loss, acc)