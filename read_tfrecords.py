import tensorflow as tf
import cv2

tf.enable_eager_execution()

def _parse_function(example_proto):
    # Create a dictionary of features.
    features = {
        'prev_img': tf.FixedLenFeature([], tf.string),
        'curr_img': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.float32)
    }

    # Parse the input tf.Example proto using the dictionary above.

    output = tf.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(output['curr_img'])

    def aug_hue(img):
        return tf.image.random_hue(img, max_delta=0.2)

    def aug_brightness(img):
        return tf.image.random_brightness(img, max_delta=0.2)

    def aug_contrast(img):
        return tf.image.random_contrast(img, 0.7, 1.3)

    def aug_saturation(img):
        return tf.image.random_saturation(img, 0.7, 1.3)

    def aug_jpg_quality(img):
        return tf.image.random_jpeg_quality(img, 5, 100)

    random_aug_process = [aug_brightness, aug_contrast, aug_hue, aug_jpg_quality, aug_saturation]
    random_aug_process_choice = tf.random_uniform(shape=(), minval=0, maxval=len(random_aug_process), dtype=tf.int64)
    aug_img = random_aug_process[random_aug_process_choice](img)
    return aug_img

if __name__ == '__main__':
    data_path = 'data/train_tfrecords/image_value_pairs_0.tfrecord'
    filenames = [data_path]
    dataset = tf.data.TFRecordDataset(filenames)

    for record in dataset:
        img = _parse_function(record)

        cv2.imshow('image', img.numpy()[...,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


