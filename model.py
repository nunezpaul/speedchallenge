# https://github.com/experiencor/speed-prediction/blob/master/Dashcam%20Speed%20-%20C3D.ipynb

import argparse
import os

import keras as k
import tensorflow as tf


# tf.enable_eager_execution()


def load_training_data(batch_size=64):
    data_path = 'data/train_tfrecords/image_value_pairs_{}.tfrecord'
    filenames = [data_path.format(i) for i in range(10)]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_training_function)
    dataset = dataset.repeat().batch(batch_size)
    dataset = dataset.shuffle(batch_size * 2)
    dataset = dataset.prefetch(batch_size * 2)
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
    crop_width = 112
    crop_height = 112
    crop_window = [crop_y, crop_x, crop_height, crop_width]

    prev_img = tf.image.decode_and_crop_jpeg(output['prev_img'], crop_window)
    curr_img = tf.image.decode_and_crop_jpeg(output['curr_img'], crop_window)

    random_rotation = tf.random_uniform((), 0, 4, dtype=tf.int32)

    prev_img, curr_img = tf.image.rot90(prev_img, random_rotation), tf.image.rot90(curr_img, random_rotation)
    combined_img = tf.image.random_brightness([prev_img, curr_img], max_delta=0.5)  # augment brightness
    combined_img = (tf.to_float(combined_img) - 225 / 2) / 255  # normalization step
    prev_img, curr_img = combined_img[0, :], combined_img[1, :]
    stacked_img = tf.stack([prev_img, curr_img])
    print(stacked_img)
    label = output['label']

    return stacked_img, label


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
    crop_x = 200 + (200 - 112)
    crop_y = 150 + (200 - 112)
    crop_width = 112
    crop_height = 112
    crop_window = [crop_y, crop_x, crop_height, crop_width]

    prev_img = tf.image.decode_and_crop_jpeg(output['prev_img'], crop_window)
    curr_img = tf.image.decode_and_crop_jpeg(output['curr_img'], crop_window)

    # Normalize the data
    prev_img = (tf.to_float(prev_img) - 225 / 2) / 255
    curr_img = (tf.to_float(curr_img) - 225 / 2) / 255

    # Combine the two images
    concated_img = tf.stack([prev_img, curr_img], 2)

    return concated_img, output['label']


def keras_model(combined_image):
    print(combined_image.shape)
    model_input = k.layers.Input(tensor=combined_image)
    conv1 = k.layers.Conv3D(64, (3, 3, 3), padding='same', name='conv1')(model_input)
    conv1 = k.layers.BatchNormalization()(conv1)
    conv1 = k.layers.Lambda(lambda x: k.layers.activations.relu(x))(conv1)
    conv1_mp = k.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(conv1)

    conv2 = k.layers.Conv3D(128, (3, 3, 3), padding='same', name='conv2')(conv1_mp)
    conv2 = k.layers.BatchNormalization()(conv2)
    conv2 = k.layers.Lambda(lambda x: k.layers.activations.relu(x))(conv2)
    conv2_mp = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(conv2)

    conv3a = k.layers.Conv3D(256, (3, 3, 3), padding='same', name='conv3a')(conv2_mp)
    conv3a = k.layers.BatchNormalization()(conv3a)
    conv3a = k.layers.Lambda(lambda x: k.layers.activations.relu(x))(conv3a)

    conv3b = k.layers.Conv3D(256, (3, 3, 3), padding='same', name='conv3b')(conv3a)
    conv3b = k.layers.BatchNormalization()(conv3b)
    conv3b = k.layers.Lambda(lambda x: k.layers.activations.relu(x))(conv3b)
    conv3_mp = k.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(conv3b)

    conv4 = k.layers.Conv3D(512, (3, 3, 3), padding='same', name='conv4a')(conv3_mp)
    conv4 = k.layers.BatchNormalization()(conv4)
    conv4 = k.layers.Lambda(lambda x: k.layers.activations.relu(x))(conv4)

    conv4b = k.layers.Conv3D(512, (3, 3, 3), padding='same', name='conv4b')(conv4)
    conv4b = k.layers.BatchNormalization()(conv4b)
    conv4b = k.layers.Lambda(lambda x: k.layers.activations.relu(x))(conv4b)
    conv4_mp = k.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(conv4b)

    conv5a = k.layers.Conv3D(512, (3, 3, 3), padding='same', name='conv5a')(conv4_mp)
    conv5a = k.layers.BatchNormalization()(conv5a)
    conv5a = k.layers.Lambda(lambda x: k.layers.activations.relu(x))(conv5a)

    conv5b = k.layers.Conv3D(512, (3, 3, 3), padding='same', name='conv5b')(conv5a)
    conv5b = k.layers.BatchNormalization()(conv5b)
    conv5b = k.layers.Lambda(lambda x: k.layers.activations.relu(x))(conv5b)

    conv5_pad = k.layers.ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5')(conv5b)
    conv5_mp = k.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(conv5_pad)

    flat = k.layers.Flatten()(conv5_mp)

    fc6 = k.layers.Dense(4096, activation='relu', name='fc6')(flat)
    fc6_dropout = k.layers.Dropout(.5)(fc6)
    fc7 = k.layers.Dense(4096, activation='relu', name='fc7')(fc6_dropout)
    fc7_dropout = k.layers.Dropout(.5)(fc7)

    diff_speed = k.layers.Dense(1, activation='linear')(fc7_dropout)
    speed = k.layers.Lambda(lambda diff_speed: diff_speed + k.backend.ones_like(diff_speed) * 12)(diff_speed)

    model = k.models.Model(inputs=model_input, outputs=speed)
    print(model.summary())

    return model


def add_average(diff_speed, avg_speed=12):
    output = k.layers.add([diff_speed, k.backend.ones_like(diff_speed) * avg_speed])
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert csv training file into tfrecord files.')
    parser.add_argument('--tpu', help='determine if to be trained on tpu', action="store_true")
    parser.add_argument('--opt', help='which optimizer to use', choices=['adam', 'sgd'])
    parser.add_argument('--lr', help='set the learning rate', type=float, default=1e-3)
    args = parser.parse_args()

    train_dataset = load_training_data()
    train_iter = train_dataset.make_one_shot_iterator()

    img, label = train_iter.get_next()
    img = tf.reshape(img, (-1, 2, 112, 112, 3))

    train_model = keras_model(img)

    if args.tpu:
        model = tf.contrib.tpu.keras_to_tpu_model(
            train_model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])))
    else:
        model = train_model

    # Compile Model
    if args.opt == 'sgd':
        opt = k.optimizers.SGD(lr=args.lr, momentum=0.9)
    elif args.opt == 'adam':
        opt = k.optimizers.adam(lr=args.lr)
    else:
        print('Need to specify your optimizer.')
        exit()
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  target_tensors=[label],
                  )

    # Let's learn!
    for i in range(20):
        model.fit(epochs=1, steps_per_epoch=8096)
        model.save('4_img_skip_model_{}.h5'.format(i))
        train_iter = train_dataset.make_one_shot_iterator()
        img, label = train_iter.get_next()