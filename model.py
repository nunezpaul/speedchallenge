# https://github.com/experiencor/speed-prediction/blob/master/Dashcam%20Speed%20-%20C3D.ipynb

import argparse
import os

import keras as k
import tensorflow as tf


# tf.enable_eager_execution()


def load_training_data():
    data_path = 'data/train_tfrecords/image_value_pairs_{}.tfrecord'
    filenames = [data_path.format(i) for i in range(10)]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_training_function)
    dataset = dataset.repeat().batch(32)
    dataset = dataset.shuffle(32)
    dataset = dataset.prefetch(100)
    # iterator = dataset.make_initializable_iterator()

    return dataset

def load_validation_data():
    data_path = 'data/val_tfrecords/val_image_value_pairs.tfrecord'
    dataset = tf.data.TFRecordDataset([data_path])
    dataset = dataset.map(_parse_val_function)
    dataset = dataset.repeat().batch(16)

    return dataset


def _parse_training_function(example_proto):
    # Dictionary of features.
    feature = {
        'prev_img': tf.FixedLenFeature([], tf.string),
        'curr_img': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.float32)
    }

    # Parse the input tf.Example proto using the dictionary above.
    output = tf.parse_single_example(example_proto, feature)

    # Randomly crop the images within a limit
    crop_x = tf.random_uniform((), 100, 340, dtype=tf.int32)
    crop_y = tf.random_uniform((), 100, 180, dtype=tf.int32)
    crop_width = 200
    crop_height = 200
    crop_window = [crop_y, crop_x, crop_height, crop_width]

    prev_img = tf.image.decode_and_crop_jpeg(output['prev_img'], crop_window)
    curr_img = tf.image.decode_and_crop_jpeg(output['curr_img'], crop_window)

    random_rotation = tf.random_uniform((), 0, 4, dtype=tf.int32)

    prev_img, curr_img = tf.image.rot90(prev_img, random_rotation), tf.image.rot90(curr_img, random_rotation)
    combined_img = tf.image.random_brightness([prev_img, curr_img], max_delta=0.5)  # augment brightness
    combined_img = (tf.to_float(combined_img) - 225 / 2) / 255  # normalization step
    prev_img, curr_img = combined_img[0, :], combined_img[1, :]
    concated_img = tf.concat([prev_img, curr_img], 2)

    label = output['label']

    return concated_img, label


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


def basic_model(combined_img):
    # Three convolutional layers.
    print(combined_img)
    conv_1 = tf.layers.conv2d(combined_img,
                              filters=64,
                              kernel_size=3,
                              strides=2,
                              activation=tf.nn.relu,
                              name='conv_1')
    conv_2 = tf.layers.conv2d(conv_1,
                              filters=64,
                              kernel_size=3,
                              strides=2,
                              activation=tf.nn.relu,
                              name='conv_2')
    conv_3 = tf.layers.conv2d(conv_2,
                              filters=64,
                              kernel_size=3,
                              strides=2,
                              activation=tf.nn.relu,
                              name='conv_3')

    # Four fully-connected layers.
    flat = tf.layers.flatten(conv_3)
    fc_4 = tf.layers.dense(flat, 4096, activation=tf.nn.relu, name='fc_4')
    fc_5 = tf.layers.dense(fc_4, 4096, activation=tf.nn.relu, name='fc_5')
    fc_6 = tf.layers.dense(fc_5, 4096, activation=tf.nn.relu, name='fc_6')
    fc_7 = tf.layers.dense(fc_6, 1, name='fc_7')

    train_model = k.models.Model(inputs=combined_img, outputs=fc_7)

    return fc_7


def keras_basic_model(combined_image):
    model_input = k.layers.Input(tensor=combined_image)
    conv1 = k.layers.Conv2D(filters=64,
                            kernel_size=3,
                            strides=2,
                            activation=tf.nn.relu,
                            name='conv_1')(model_input)
    conv2 = k.layers.Conv2D(filters=64,
                            kernel_size=3,
                            strides=2,
                            activation=tf.nn.relu,
                            name='conv_2')(conv1)
    conv3 = k.layers.Conv2D(filters=64,
                            kernel_size=3,
                            strides=2,
                            activation=tf.nn.relu,
                            name='conv_3')(conv2)
    flat = k.layers.Flatten()(conv3)
    fully_connected1 = k.layers.Dense(4096, activation=tf.nn.relu, name='fc_4')(flat)
    fully_connected2 = k.layers.Dense(4096, activation=tf.nn.relu, name='fc_5')(fully_connected1)
    fully_connected3 = k.layers.Dense(4096, activation=tf.nn.relu, name='fc_6')(fully_connected2)
    model_output = k.layers.Dense(1)(fully_connected3)

    train_model = k.models.Model(inputs=model_input, outputs=model_output)

    print(train_model.summary())

    return train_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert csv training file into tfrecord files.')
    parser.add_argument('--tpu', help='determine if to be trained on tpu', action="store_true")
    args = parser.parse_args()

    train_dataset = load_training_data()
    train_iter = train_dataset.make_one_shot_iterator()

    img, label = train_iter.get_next()
    img = tf.reshape(img, (-1, 200, 200, 6))
    train_model = keras_basic_model(img)

    if args.tpu:
        model = tf.contrib.tpu.keras_to_tpu_model(
            train_model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])))
    else:
        model = train_model

    # Compile Model
    model.compile(optimizer=k.optimizers.adam(),
                  loss='mean_squared_error',
                  target_tensors=[label],
                  )

    # Let's learn!
    for i in range(20):
        model.fit(lr=0.01, epochs=1, steps_per_epoch=8096)
        model.save('4_img_skip_model_{}.h5'.format(i))
        train_iter = train_dataset.make_one_shot_iterator()
        img, label = train_iter.get_next()