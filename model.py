import os
import uuid

import tensorflow as tf

from config import Config

# tf.enable_eager_execution()


class DataBase(object):
    def __init__(self):
        # Feature dictionary to pull from TFRecord.
        self.features = {
            'prev_img': tf.FixedLenFeature([], tf.string),
            'curr_img': tf.FixedLenFeature([], tf.string),
            'category': tf.FixedLenFeature([], tf.int64),
            'speed': tf.FixedLenFeature([], tf.float32)
        }

        # Image crop dimensions
        self.crop_width = 640
        self.crop_height = 300

        # Combined total num channels
        self.num_channels = 6

    def setup_dataset_iter(self, filenames, batch_size, _parse_function):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)
        dataset = dataset.repeat().batch(batch_size)
        # dataset = dataset.shuffle(batch_size)
        dataset = dataset.prefetch(batch_size * 2)
        iterator = dataset.make_one_shot_iterator()
        img, label, speed = iterator.get_next()
        return img, label, speed, iterator

    def normalize_img(self, img):
        return (tf.to_float(img) - 225. / 2.) / 255.


class TrainData(DataBase):
    def __init__(self, file, num_shards, batch_size):
        super(TrainData, self).__init__()

        # Online data augmentation values
        self.max_random_hue_delta = 0.2
        self.max_random_brightness_delta = 0.2
        self.random_contrast_range = [0.7, 1.3]
        self.random_saturation_range = [0.7, 1.3]
        self.random_jpeg_quality_range = [5, 100]

        # Finish setting up the dataset
        filenames = [file.format(i) for i in range(num_shards)]
        self.img, self.label, self.speed, self.iter = self.setup_dataset_iter(filenames, batch_size, self._parse_function)
        self.img = tf.reshape(self.img, (-1, 300, 640, 6))

    def _parse_function(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        output = tf.parse_single_example(example_proto, self.features)

        # Grab each both images from example proto
        prev_img = tf.image.decode_jpeg(output['prev_img'])
        curr_img = tf.image.decode_jpeg(output['curr_img'])

        # Stack the images by their channel and apply the same random crop
        stacked_img = tf.concat([prev_img, curr_img], axis=-1)
        stacked_img = tf.image.random_crop(stacked_img, [self.crop_height, self.crop_width, self.num_channels])

        # Split the data and restack them (side-by-side) to apply the same random image processing
        side_by_side_img = tf.concat([stacked_img[:, :, :3], stacked_img[:, :, 3:]], 0)

        # Selecting one random augmentations (or none) to apply to the image pairs
        aug_process = tf.random_uniform(shape=(), minval=0, maxval=6, dtype=tf.int64)
        side_by_side_img = tf.cond(tf.less(aug_process, 2),
                                   lambda: self._true_fn_l1(side_by_side_img, aug_process),
                                   lambda: self._false_fn_l1(side_by_side_img, aug_process))

        # Randomly flip the images vertically or horizontally
        side_by_side_img = tf.image.random_flip_left_right(side_by_side_img)
        side_by_side_img = tf.image.random_flip_up_down(side_by_side_img)

        # Restack the imgs by their channel
        stacked_img = tf.concat([side_by_side_img[:self.crop_height, :, :],
                                 side_by_side_img[self.crop_height:, :, :]],
                                axis=-1)

        # Normalize the img data
        stacked_img = self.normalize_img(stacked_img)

        # Add random gaussian noise to the images
        # stacked_img = stacked_img + tf.random_normal(stddev=0.5,
        #                                              shape=[self.crop_height,
        #                                                     self.crop_width,
        #                                                     self.num_channels],
        #                                              )

        return stacked_img, output['category'], output['speed']

    # This section determines which random function for data augmentation is applied while training
    def _true_fn_l1(self, img, val):
        # if random val == 0 or 1
        def aug_hue(img, max_hue_delta):
            return tf.image.random_hue(img, max_delta=max_hue_delta)

        def aug_brightness(img, max_brightness_delta):
            return tf.image.random_brightness(img, max_delta=max_brightness_delta)

        return tf.cond(tf.equal(val, 0),
                       lambda: aug_hue(img, self.max_random_hue_delta),  # if val is 0
                       lambda: aug_brightness(img, self.max_random_brightness_delta))  # if val is 1

    def _false_fn_l1(self, img, val):
        # if random val >= 2
        return tf.cond(tf.less(val, 4),
                       lambda: self._true_fn_l2(img, val),  # if val is 2 or 3
                       lambda: self._false_fn_l2(img, val))  # if val is 4 or 5

    def _true_fn_l2(self, img, val):
        # if random val == 2 or 3
        def aug_contrast(img, random_contrast_range):
            return tf.image.random_contrast(img, *random_contrast_range)

        def aug_saturation(img, random_saturation_range):
            return tf.image.random_saturation(img, *random_saturation_range)

        return tf.cond(tf.equal(val, 2),
                       lambda: aug_contrast(img, self.random_contrast_range),  # if val is 2
                       lambda: aug_saturation(img, self.random_saturation_range))  # if val is 3

    def _false_fn_l2(self, img, val):
        # if random val == 4 or 5
        def aug_jpg_quality(img, random_jpeg_quality_range):
            return tf.image.random_jpeg_quality(img, *random_jpeg_quality_range)

        return tf.cond(tf.equal(val, 4),
                       lambda: aug_jpg_quality(img, self.random_jpeg_quality_range),  # if val is 4
                       lambda: img)  # if val is 5


class ValidData(DataBase):
    def __init__(self, file, batch_size):
        super(ValidData, self).__init__()
        self.crop_x_left = 0
        self.crop_y_top = 50

        # Finish setting up the dataset iterator
        self.img, self.label, self.speed, self.iter = self.setup_dataset_iter([file], batch_size, self._parse_function)

    def _parse_function(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        output = tf.parse_single_example(example_proto, self.features)

        # Crop window for validation data
        crop_window = [self.crop_y_top, self.crop_x_left, self.crop_height, self.crop_width]

        # Crop a fixed region of the images
        prev_img = tf.image.decode_and_crop_jpeg(output['prev_img'], crop_window)
        curr_img = tf.image.decode_and_crop_jpeg(output['curr_img'], crop_window)

        # Stack the two images and normalize them
        stacked_img = tf.concat([prev_img, curr_img], axis=-1)
        stacked_img = self.normalize_img(stacked_img)

        return stacked_img, tf.clip_by_value(output['category'], 0, 9), output['speed']
    

class DeepVO(object):
    def __init__(self, train_data, dropout, bucket_size, load_model, opt, lr, tpu, save_dir):
        self.save_dir = save_dir if save_dir else ''
        self.uuid = uuid.uuid4()
        self.load_model = load_model
        self.dropout = dropout
        self.bucket_size = bucket_size
        self.num_buckets = 30 // bucket_size
        self.optimizer = self.setup_optimizer(opt, lr)
        self.callbacks = self.setup_callbacks()
        self.model = self.setup_model(train_data)

        #convert model to tpu model
        if tpu:
            tpu_worker = 'grpc://' + os.environ['COLAB_TPU_ADDR']
            strategy = tf.contrib.tpu.TPUDistributionStrategy(
                    tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_worker))
            self.model = tf.contrib.tpu.keras_to_tpu_model(self.model, strategy=strategy)

    def setup_optimizer(self, opt, lr):
        if opt == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        else:
            opt = tf.train.AdamOptimizer(learning_rate=lr)
        return opt

    def setup_callbacks(self):
        callbacks = []
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.save_dir + f'./log/{self.uuid}',
                                                     histogram_freq=0,
                                                     write_graph=True,
                                                     write_images=True)
        callbacks.append(tensorboard)
        return callbacks

    def setup_model(self, train_data):
        if self.load_model:
            model = tf.keras.models.load_model(self.load_model)
        else:
            model_input = tf.keras.layers.Input(tensor=train_data.img)
            model_output = self.cnn(model_input)
            model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

            model.compile(optimizer=self.optimizer,
                          loss=self.sparse_categorical_crossentropy,
                          target_tensors=[train_data.speed],
                          metrics=[self.categorical_accuracy, self.mean_squared_error],
                          )
            print(model.summary())

        return model

    def cnn(self, model_input):
        conv1 = tf.keras.layers.ZeroPadding2D((3, 3))(model_input)
        conv1 = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=2, padding='valid', name='conv1')(conv1)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x), name='relu1')(conv1)

        conv2 = tf.keras.layers.ZeroPadding2D((2, 2))(conv1)
        conv2 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=2, padding='valid', name='conv2')(conv2)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x), name='relu2')(conv2)

        conv3a = tf.keras.layers.ZeroPadding2D((2, 2))(conv2)
        conv3a = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=2, padding='valid', name='conv3a')(conv3a)
        conv3a = tf.keras.layers.BatchNormalization()(conv3a)
        conv3a = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x), name='relu3a')(conv3a)

        conv3b = tf.keras.layers.ZeroPadding2D()(conv3a)
        conv3b = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='valid', name='conv3b')(conv3b)
        conv3b = tf.keras.layers.BatchNormalization()(conv3b)
        conv3b = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x), name='relu3b')(conv3b)

        conv4 = tf.keras.layers.ZeroPadding2D()(conv3b)
        conv4 = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=2, padding='valid', name='conv4a')(conv4)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x), name='relu4')(conv4)

        conv4b = tf.keras.layers.ZeroPadding2D()(conv4)
        conv4b = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='valid', name='conv4b')(conv4b)
        conv4b = tf.keras.layers.BatchNormalization()(conv4b)
        conv4b = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x), name='relu4b')(conv4b)

        conv5a = tf.keras.layers.ZeroPadding2D()(conv4b)
        conv5a = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=2, padding='valid', name='conv5a')(conv5a)
        conv5a = tf.keras.layers.BatchNormalization()(conv5a)
        conv5a = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x), name='relu5a')(conv5a)

        conv5b = tf.keras.layers.ZeroPadding2D()(conv5a)
        conv5b = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='valid', name='conv5b')(conv5b)
        conv5b = tf.keras.layers.BatchNormalization()(conv5b)
        conv5b = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x), name='relu5b')(conv5b)

        conv6 = tf.keras.layers.ZeroPadding2D()(conv5b)
        conv6 = tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=2, padding='valid', name='conv6')(conv6)
        conv6 = tf.keras.layers.BatchNormalization()(conv6)
        conv6 = tf.keras.layers.MaxPool2D(pool_size=(2, 3))(conv6)  # only difference in addition from original paper

        flat = tf.keras.layers.Flatten()(conv6)

        fc6 = tf.keras.layers.Dense(1024, activation='relu', name='fc6')(flat)
        fc6_dropout = tf.keras.layers.Dropout(self.dropout)(fc6)

        fc7 = tf.keras.layers.Dense(1024, activation='relu', name='fc7')(fc6_dropout)
        fc7_dropout = tf.keras.layers.Dropout(self.dropout)(fc7)

        speed_prob = tf.keras.layers.Dense(self.num_buckets, activation='softmax')(fc7_dropout)

        return speed_prob

    def mean_squared_error(self, ys, y_pred):
        y_true = ys
        y_pred = tf.argmax(y_pred, -1)
        y_pred_fp = (tf.to_float(y_pred) + 0.5) * self.bucket_size

        return tf.reduce_mean(tf.square(y_true - y_pred_fp))

    def sparse_categorical_crossentropy(self, ys, y_pred):
        y_true = tf.clip_by_value(ys // self.bucket_size, 0, self.num_buckets - 1)
        cat_crossentropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return cat_crossentropy_loss

    def categorical_accuracy(self, ys, y_pred):
        y_true = tf.clip_by_value(ys // self.bucket_size, 0, self.num_buckets - 1)
        y_pred = tf.argmax(y_pred, -1)
        same_cat = tf.equal(tf.to_int64(y_true), y_pred)
        return tf.reduce_mean(tf.to_float(same_cat))

    def fit(self, train_data, valid_data):
        for i in range(20):
            self.model.fit(
                           epochs=5,
                           steps_per_epoch=20000 // 32,
                           validation_data=[valid_data.img, valid_data.speed] if valid_data else None,
                           validation_steps=63,
                           callbacks=self.callbacks,)
            self.model.save(self.save_dir + f'speed_model_{self.optimizer}_{i}.h5')


if __name__ == '__main__':
    config = Config()
    train_data = TrainData('data/tfrecords/train/shard_{}.tfrecord', num_shards=10, batch_size=32)
    valid_data = ValidData('data/tfrecords/val/val.tfrecord', batch_size=32)

    deep_vo = DeepVO(train_data=train_data, **config.params)

    if config.params['save_dir']:
        from google.colab import drive
        drive.mount('gdrive')

    deep_vo.fit(train_data=train_data, valid_data=valid_data)