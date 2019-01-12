import tensorflow as tf

from multiprocessing import cpu_count


# tf.enable_eager_execution()


class DataBase(object):
    def __init__(self, len, batch_size, training=False):
        # Save information about the dataset for processing later.
        self.training = training
        self.len = len
        self.batch_size = batch_size

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
        self.img_shape = (self.batch_size, self.crop_height, self.crop_width, self.num_channels)

    def __len__(self):
        return self.len

    def setup_dataset_iter(self, filenames, _parse_function):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function, num_parallel_calls=cpu_count())
        dataset = dataset.repeat().batch(self.batch_size)
        # dataset = dataset.shuffle(batch_size)
        dataset = dataset.prefetch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()

        # Output will be img, label and speed for train and val data but just img for test data
        get_next_output = list(iterator.get_next())

        # First position will always be the image. Reshaping here.
        get_next_output[0] = tf.reshape(get_next_output[0], self.img_shape)

        # If there are additional params then they are reshaped to include the batch size
        for i in range(1, len(get_next_output)):
            get_next_output[i] = tf.reshape(get_next_output[i], (self.batch_size,))

        return get_next_output + [iterator]

    def per_image_standardization(self, images):
        images = tf.cast(images, dtype=tf.float32)
        images_mean = tf.reduce_mean(images, axis=[1, 2, 3], keepdims=True)

        variances = tf.reduce_mean(tf.square(images), axis=[1, 2, 3], keepdims=True) - tf.square(images_mean)
        variances = tf.nn.relu(variances)
        stddevs = tf.sqrt(variances)

        # Minimum normalization that protects against uniform images
        num_pixels_per_img = tf.reduce_prod(tf.shape(images)[1:])
        min_stddevs = tf.rsqrt(tf.cast(num_pixels_per_img, dtype=tf.float32))

        pixel_value_scale = tf.maximum(stddevs, min_stddevs)
        pixel_value_offset = images_mean

        images = tf.subtract(images, pixel_value_offset)
        images = tf.div(images, pixel_value_scale)

        return images


class TrainData(DataBase):
    def __init__(self, file, num_shards, batch_size, len, training, class_weights_csv=None):
        super(TrainData, self).__init__(batch_size=batch_size, len=len * num_shards, training=training)
        # Determine class weights
        self.class_weights = self._get_class_weights(class_weights_csv)

        # Online data augmentation values
        self.max_random_hue_delta = 0.4
        self.max_random_brightness_delta = 0.4
        self.random_contrast_range = [0.6, 1.4]
        self.random_saturation_range = [0.6, 1.4]
        self.random_jpeg_quality_range = [5, 100]

        # Finish setting up the dataset
        filenames = [file.format(i) for i in range(num_shards)]
        self.img, self.label, self.speed, self.iter = self.setup_dataset_iter(filenames, self._parse_function)

        # Add gaussian noise to the images and clip it to the range of the img [0, 1)

        self.img = self.img + tf.random_normal(stddev=0.5, shape=self.img_shape)
        self.img = tf.clip_by_value(self.img, clip_value_min=0.0, clip_value_max=1.0)

        # Normalize the images
        self.img = self.per_image_standardization(self.img)

    def _get_class_weights(self, class_weights_csv):
        if not class_weights_csv:
            return

        with open(class_weights_csv) as f:
            weights = f.read().splitlines()

        weights = [float(val) for val in weights]
        class_weights = dict(zip(list(range(len(weights))), weights))
        return class_weights

    def _parse_function(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        output = tf.parse_single_example(example_proto, self.features)

        # Grab each both images from example proto
        prev_img = tf.image.decode_jpeg(output['prev_img'])
        curr_img = tf.image.decode_jpeg(output['curr_img'])

        # Stack the images by their channel and apply the same random crop
        stacked_img = tf.concat([prev_img, curr_img], axis=-1)
        stacked_img = tf.image.random_crop(stacked_img, [self.crop_height, self.crop_width, self.num_channels])
        stacked_img = tf.to_float(stacked_img) / 255.  # rescaling [0 1)

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

        labels = []
        if 'category' in output:
            labels.append(output['category'])
            labels.append(output['speed'])

        return [stacked_img] + labels

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


class NonTrainData(DataBase):
    def __init__(self, batch_size, len):
        super(NonTrainData, self).__init__(batch_size=batch_size, len=len)
        self.crop_x_left = 0
        self.crop_y_top = 50

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
        stacked_img = tf.to_float(stacked_img) / 255.

        labels = []
        if 'category' in output:
            labels.append(tf.clip_by_value(output['category'], 0, 9))
            labels.append(output['speed'])

        return [stacked_img] + labels


class ValidData(NonTrainData):
    def __init__(self, file, batch_size, len):
        super(ValidData, self).__init__(batch_size=batch_size, len=len)
        # Finish setting up the dataset iterator
        self.img, self.label, self.speed, self.iter = self.setup_dataset_iter([file], self._parse_function)
        self.img = self.per_image_standardization(self.img)


class TestData(NonTrainData):
    def __init__(self, file, batch_size, len):
        super(TestData, self).__init__(batch_size=batch_size, len=len)

        # Feature dictionary to pull from TFRecord.
        self.features = {
            'prev_img': tf.FixedLenFeature([], tf.string),
            'curr_img': tf.FixedLenFeature([], tf.string),
        }
        self.img, self.iter = self.setup_dataset_iter([file], self._parse_function)
        self.img = self.per_image_standardization(self.img)


if __name__ == '__main__':
    valid_data = ValidData('data/tfrecords/val/val.tfrecord', batch_size=32, len=8615)
    test_data = TestData('data/tfrecords/test/test.tfrecord', batch_size=32, len=10797)
    train_data = TrainData('data/tfrecords/train/train.tfrecord',
                           num_shards=1,
                           batch_size=32,
                           len=2000,
                           training=True,
                           class_weights_csv='data/labeled_csv/train/train_class_weights.csv')

    for data in (train_data, valid_data, test_data):
        # check image size
        print(data.img.shape, data.img_shape)
        assert data.img.shape == (data.batch_size, data.crop_height, data.crop_width, data.num_channels)

    for data in (train_data, valid_data):
        # check category size
        print(data.label.shape)
        assert data.label.shape == (data.batch_size,)

        # check speed size
        print(data.speed.shape)
        assert data.speed.shape == (data.batch_size,)

    test_img = tf.random_uniform((32, 400, 600, 6))
    my_test_img_normed = train_data.per_image_standardization(test_img)
    tf_test_img_normed = tf.map_fn(tf.image.per_image_standardization, test_img)
    difference = tf.reduce_mean(tf.abs(tf.subtract(my_test_img_normed, tf_test_img_normed)))
    print(difference)
