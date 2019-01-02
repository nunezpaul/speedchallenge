import tensorflow as tf
import cv2

tf.enable_eager_execution()


# if random val == 0 or 1
def cond1_true(img, val):
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


def _parse_function(example_proto):
    # Create a dictionary of features.
    features = {
        'prev_img': tf.FixedLenFeature([], tf.string),
        'curr_img': tf.FixedLenFeature([], tf.string),
        # 'category': tf.FixedLenFeature([], tf.int64),
        # 'speed': tf.FixedLenFeature([], tf.float32)
    }

    # Parse the input tf.Example proto using the dictionary above.

    output = tf.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(output['curr_img'])

    aug_process = tf.random_uniform(shape=(), minval=0, maxval=6, dtype=tf.int64)
    aug_img = tf.cond(tf.less(aug_process, 2),
                      lambda: cond1_true(img, aug_process),
                      lambda: cond1_false(img, aug_process))
    return aug_img

if __name__ == '__main__':
    data_path = 'data/tfrecords/test/test.tfrecord'
    filenames = [data_path]
    dataset = tf.data.TFRecordDataset(filenames)

    for record in dataset:
        img = _parse_function(record)

        cv2.imshow('image', img.numpy()[...,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


