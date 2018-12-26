# https://github.com/experiencor/speed-prediction/blob/master/Dashcam%20Speed%20-%20C3D.ipynb

import argparse
import os
import uuid

import keras as k
import tensorflow as tf


# tf.enable_eager_execution()


def load_training_data(batch_size=64):
    data_path = 'data/train_tfrecords/image_value_pairs_{}.tfrecord'
    filenames = [data_path.format(i) for i in range(10)]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_training_function)
    dataset = dataset.repeat().batch(batch_size)
    # dataset = dataset.shuffle(batch_size)
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
    crop_x = 0
    crop_y = tf.random_uniform((), 0, 50, dtype=tf.int32)
    crop_width = 640
    crop_height = 300
    crop_window = [crop_y, crop_x, crop_height, crop_width]

    prev_img = tf.image.decode_and_crop_jpeg(output['prev_img'], crop_window)
    curr_img = tf.image.decode_and_crop_jpeg(output['curr_img'], crop_window)
    stacked_img = tf.concat([prev_img, curr_img], axis=-1)

    stacked_img = tf.image.random_flip_left_right(stacked_img)
    stacked_img = tf.image.random_flip_up_down(stacked_img)
    stacked_img = tf.image.random_brightness(stacked_img, max_delta=0.5)  # augment brightness
    stacked_img = (tf.to_float(stacked_img) - 225 / 2) / 255.  # normalization step

    label = output['label']
    cat_label = tf.to_int32(label)

    return stacked_img, cat_label


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


def keras_model(model_input):
    conv1 = k.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='valid', name='conv1')(model_input)
    conv1 = k.layers.BatchNormalization()(conv1)
    conv1 = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu1')(conv1)

    conv2 = k.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='valid', name='conv2')(conv1)
    conv2 = k.layers.BatchNormalization()(conv2)
    conv2 = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu2')(conv2)

    conv3a = k.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='valid', name='conv3a')(conv2)
    conv3a = k.layers.BatchNormalization()(conv3a)
    conv3a = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu3a')(conv3a)

    flat = k.layers.Flatten()(conv3a)

    fc6 = k.layers.Dense(4096, activation='relu', name='fc6')(flat)
    fc6_dropout = k.layers.Dropout(.5)(fc6)

    fc7 = k.layers.Dense(4096, activation='relu', name='fc7')(fc6_dropout)
    fc7_dropout = k.layers.Dropout(.5)(fc7)

    fc8 = k.layers.Dense(4096, activation='relu', name='fc8')(fc7_dropout)
    fc8_dropout = k.layers.Dropout(.5)(fc8)

    speed_prob = k.layers.Dense(10, activation='softmax')(fc8_dropout)

    model = k.models.Model(inputs=model_input, outputs=speed_prob)
    print(model.summary())

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert csv training file into tfrecord files.')
    parser.add_argument('--tpu', help='determine if to be trained on tpu', action="store_true")
    parser.add_argument('--opt', help='which optimizer to use', choices=['adam', 'sgd'])
    parser.add_argument('--lr', help='set the learning rate', type=float, default=1e-3)
    args = parser.parse_args()

    train_dataset = load_training_data()
    train_iter = train_dataset.make_one_shot_iterator()

    img, label = train_iter.get_next()
    img = tf.reshape(img, (-1, 300, 640, 6))

    print(img.shape)
    model_input = k.layers.Input(tensor=img)
    train_model = keras_model(model_input)

    if args.tpu:
        model = tf.contrib.tpu.keras_to_tpu_model(
            k.Model(model_input, k.layers.Lambda(lambda x: train_model(x))(model_input)),
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])))
    else:
        model = train_model

    # Compile Model
    if args.opt == 'sgd':
        opt = k.optimizers.SGD(lr=args.lr)#, momentum=0.9)
    elif args.opt == 'adam':
        opt = k.optimizers.adam(lr=args.lr)
    else:
        print('Need to specify your optimizer.')
        opt = k.optimizers.SGD(lr=args.lr)
    print(label)
    uuid = uuid.uuid4()
    callbacks = k.callbacks.TensorBoard(log_dir=f'./log/{uuid}', histogram_freq=0, write_graph=True, write_images=True)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  target_tensors=[label],
                  metrics=['categorical_accuracy']
                  )

    # Let's learn!
    for i in range(20):
        model.fit(epochs=1, steps_per_epoch=20000//64)
        model.save('4_img_skip_model_{}.h5'.format(i))
        train_iter = train_dataset.make_one_shot_iterator()
        img, label = train_iter.get_next()