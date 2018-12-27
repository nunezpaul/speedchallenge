# https://github.com/experiencor/speed-prediction/blob/master/Dashcam%20Speed%20-%20C3D.ipynb

import argparse
import os
import uuid

import keras as k
import tensorflow as tf


# tf.enable_eager_execution()


def load_training_data(batch_size=32):
    data_path = 'data/train_tfrecords/image_value_pairs_{}.tfrecord'
    filenames = [data_path.format(i) for i in range(10)]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_training_function)
    dataset = dataset.repeat().batch(batch_size)
    # dataset = dataset.shuffle(batch_size)
    dataset = dataset.prefetch(batch_size * 2)
    # iterator = dataset.make_initializable_iterator()

    return dataset


def load_validation_data(batch_size=32):
    data_path = 'data/val_tfrecords/val_image_value_pairs.tfrecord'
    dataset = tf.data.TFRecordDataset([data_path])
    dataset = dataset.map(_parse_val_function)
    dataset = dataset.repeat().batch(batch_size)

    return dataset


def _parse_training_function(example_proto):
    # if random val == 0 or 1
    def cond1_true(img, val):
        print('cond1 True', val)

        def aug_hue(img):
            return tf.image.random_hue(img, max_delta=0.2)

        def aug_brightness(img):
            return tf.image.random_brightness(img, max_delta=0.2)

        return tf.cond(tf.equal(val, 0),
                       lambda: aug_hue(img),
                       lambda: aug_brightness(img))

    # if random val > 1
    def cond1_false(img, val):
        # if random val == 2 or 3
        def cond2_true(img, val):
            def aug_contrast(img):
                return tf.image.random_contrast(img, 0.7, 1.3)

            def aug_saturation(img):
                return tf.image.random_saturation(img, 0.7, 1.3)

            return tf.cond(tf.equal(val, 2), lambda: aug_contrast(img), lambda: aug_saturation(img))

        # if random val == 4 or 5
        def cond2_false(img, val):
            def aug_jpg_quality(img):
                return tf.image.random_jpeg_quality(img, 5, 100)

            return tf.cond(tf.equal(val, 5),
                           lambda: aug_jpg_quality(img),
                           lambda: img)

        return tf.cond(tf.less(val, 5),
                       lambda: cond2_true(img, val),
                       lambda: cond2_false(img, val))

    # Dictionary of features.
    feature = {
        'prev_img': tf.FixedLenFeature([], tf.string),
        'curr_img': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.float32)
    }

    # Parse the input tf.Example proto using the dictionary above.
    output = tf.parse_single_example(example_proto, feature)

    # Randomly crop the images within a limit
    crop_width = 640
    crop_height = 300
    num_channels = 6

    # Grab each img then apply the same random crop
    prev_img = tf.image.decode_jpeg(output['prev_img'])
    curr_img = tf.image.decode_jpeg(output['curr_img'])
    stacked_img = tf.concat([prev_img, curr_img], axis=-1)
    stacked_img = tf.image.random_crop(stacked_img, [crop_height, crop_width, num_channels])

    # Split the data and restack them to apply the same random image processing
    side_by_side_img = tf.concat([stacked_img[:, :, :3], stacked_img[:, :, 3:]], 0)

    # Selecting one random augmentations (or none) to apply to the image pairs
    aug_process = tf.random_uniform(shape=(), minval=0, maxval=6, dtype=tf.int64)
    side_by_side_img = tf.cond(tf.less(aug_process, 2),
                               lambda: cond1_true(side_by_side_img, aug_process),
                               lambda: cond1_false(side_by_side_img, aug_process))

    # Randomly flip the images vertically or horizontally
    side_by_side_img = tf.image.random_flip_left_right(side_by_side_img)
    side_by_side_img = tf.image.random_flip_up_down(side_by_side_img)

    # Restack the imgs by their channel
    stacked_img = tf.concat([side_by_side_img[:crop_height, :, :], side_by_side_img[crop_height:, :, :]], axis=-1)

    # Normalize the img data and add random noise
    stacked_img = (tf.to_float(stacked_img) - 225 / 2) / 255.
    # stacked_img = stacked_img + tf.random_normal(shape=[crop_height, crop_width, num_channels], stddev=0.5)

    label = output['label']
    cat_label = tf.to_int64(label)

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
    crop_x = 0
    crop_y = 50
    crop_width = 640
    crop_height = 300
    crop_window = [crop_y, crop_x, crop_height, crop_width]

    # Crop the center images
    prev_img = tf.image.decode_and_crop_jpeg(output['prev_img'], crop_window)
    curr_img = tf.image.decode_and_crop_jpeg(output['curr_img'], crop_window)

    # Stack the two images and normalize them
    stacked_img = tf.concat([prev_img, curr_img], axis=-1)
    stacked_img = (tf.to_float(stacked_img) - 225 / 2) / 255.

    label = output['label']
    cat_label = tf.to_int64(label)

    return stacked_img, cat_label


def keras_model(combined_image):
    print(combined_image.shape)
    model_input = k.layers.Input(tensor=combined_image)
    conv1 = k.layers.ZeroPadding2D((3, 3))(model_input)
    conv1 = k.layers.Conv2D(64, kernel_size=(7, 7), strides=2, padding='valid', name='conv1')(conv1)
    conv1 = k.layers.BatchNormalization()(conv1)
    conv1 = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu1')(conv1)

    conv2 = k.layers.ZeroPadding2D((2, 2))(conv1)
    conv2 = k.layers.Conv2D(128, kernel_size=(5, 5), strides=2, padding='valid', name='conv2')(conv2)
    conv2 = k.layers.BatchNormalization()(conv2)
    conv2 = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu2')(conv2)

    conv3a = k.layers.ZeroPadding2D((2, 2))(conv2)
    conv3a = k.layers.Conv2D(256, kernel_size=(5, 5), strides=2, padding='valid', name='conv3a')(conv3a)
    conv3a = k.layers.BatchNormalization()(conv3a)
    conv3a = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu3a')(conv3a)

    conv3b = k.layers.ZeroPadding2D()(conv3a)
    conv3b = k.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='valid', name='conv3b')(conv3b)
    conv3b = k.layers.BatchNormalization()(conv3b)
    conv3b = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu3b')(conv3b)

    conv4 = k.layers.ZeroPadding2D()(conv3b)
    conv4 = k.layers.Conv2D(512, kernel_size=(3, 3), strides=2, padding='valid', name='conv4a')(conv4)
    conv4 = k.layers.BatchNormalization()(conv4)
    conv4 = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu4')(conv4)

    conv4b = k.layers.ZeroPadding2D()(conv4)
    conv4b = k.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='valid', name='conv4b')(conv4b)
    conv4b = k.layers.BatchNormalization()(conv4b)
    conv4b = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu4b')(conv4b)

    conv5a = k.layers.ZeroPadding2D()(conv4b)
    conv5a = k.layers.Conv2D(512, kernel_size=(3, 3), strides=2, padding='valid', name='conv5a')(conv5a)
    conv5a = k.layers.BatchNormalization()(conv5a)
    conv5a = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu5a')(conv5a)

    conv5b = k.layers.ZeroPadding2D()(conv5a)
    conv5b = k.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='valid', name='conv5b')(conv5b)
    conv5b = k.layers.BatchNormalization()(conv5b)
    conv5b = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu5b')(conv5b)

    conv6 = k.layers.ZeroPadding2D()(conv5b)
    conv6 = k.layers.Conv2D(1024, kernel_size=(3, 3), strides=2, padding='valid', name='conv6')(conv6)
    conv6 = k.layers.BatchNormalization()(conv6)
    conv6 = k.layers.MaxPool2D(pool_size=(2, 3))(conv6)

    flat = k.layers.Flatten()(conv6)

    fc6 = k.layers.Dense(1024, activation='relu', name='fc6')(flat)
    fc6_dropout = k.layers.Dropout(.5)(fc6)

    fc7 = k.layers.Dense(1024, activation='relu', name='fc7')(fc6_dropout)
    fc7_dropout = k.layers.Dropout(.5)(fc7)

    speed_prob = k.layers.Dense(10, activation='softmax')(fc7_dropout)

    model = k.models.Model(inputs=model_input, outputs=speed_prob)
    print(model.summary())

    return model


def mse_metric(y_true, y_pred, bucket_size=3):
    y_pred = tf.argmax(y_pred, -1)
    y_true_fp = (tf.to_float(y_true) + 0.5) * bucket_size
    y_pred_fp = (tf.to_float(y_pred) + 0.5) * bucket_size

    return tf.reduce_mean(tf.square(y_true_fp - y_pred_fp))


def categorical_accuracy(y_true, y_pred):
    y_pred = tf.argmax(y_pred, -1)
    same_cat = tf.equal(y_true, y_pred)
    return tf.reduce_mean(tf.to_float(same_cat))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert csv training file into tfrecord files.')
    parser.add_argument('--tpu', help='determine if to be trained on tpu', action="store_true")
    parser.add_argument('--opt', help='which optimizer to use', choices=['adam', 'sgd'])
    parser.add_argument('--lr', help='set the learning rate', type=float, default=1e-3)
    args = parser.parse_args()

    # Creating train data iterator
    train_dataset = load_training_data()
    train_iter = train_dataset.make_one_shot_iterator()

    # Creating validation data iterator
    valid_dataset = load_validation_data()
    valid_iter = valid_dataset.make_one_shot_iterator()
    X_valid, y_valid = valid_iter.get_next()

    img, label = train_iter.get_next()
    img = tf.reshape(img, (-1, 300, 640, 6))

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
        opt = k.optimizers.adam(lr=args.lr)
    print(label)
    uuid = uuid.uuid4()
    callbacks = k.callbacks.TensorBoard(log_dir=f'./log/{uuid}', histogram_freq=0, write_graph=True, write_images=True)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  target_tensors=[label],
                  metrics=[categorical_accuracy, mse_metric]
                  )

    # Let's learn!
    for i in range(20):
        model.fit(train_iter, epochs=5, steps_per_epoch=20000//32,
                  validation_data=[X_valid, y_valid], validation_steps=32)
        model.save(f'speed_model_{args.opt}_{i}.h5')
        train_iter = train_dataset.make_one_shot_iterator()
        img, label = train_iter.get_next()