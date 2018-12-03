import tensorflow as tf

tf.enable_eager_execution()

def _parse_function(example_proto):
    # Create a dictionary of features.

    features = {
        'prev_img': tf.FixedLenFeature([], tf.string),
        'curr_img': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.float32)
    }

    # Parse the input tf.Example proto using the dictionary above.

    return tf.parse_single_example(example_proto, features)

if __name__ == '__main__':
    data_path = 'data/train_tfrecords/image_value_pairs_0.tfrecord'
    filenames = [data_path]
    dataset = tf.data.TFRecordDataset(filenames)

    for record in dataset:
        output = _parse_function(record)
        image = tf.image.decode_jpeg(output['curr_img'])
        break


