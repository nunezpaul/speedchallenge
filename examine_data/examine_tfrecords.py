import tensorflow as tf
import cv2

from speedchallenge.datasets import TrainData, TestData, ValidData

tf.enable_eager_execution()


def inspect_images(data):
    images = data.img

    for idx, img in enumerate(images):
        img_0 = img[:, :, :3]
        img_1 = img[:, :, 3:]
        stacked_img = tf.concat([0.5 * img_0 + 0.5 * img_1,  img_0, img_1], axis=0)

        print(f'image {idx}')
        cv2.imshow('image', stacked_img.numpy()[..., ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.enable_eager_execution()
    train_data = TrainData('../data/tfrecords/train/train.tfrecord',
                           num_shards=1,
                           batch_size=32,
                           len=18385,
                           training=True)
    valid_data = TestData('../data/tfrecords/val/val.tfrecord', batch_size=32, len=2130)
    sorted_train_data = TestData('../data/tfrecords/val/sorted_train.tfrecord', batch_size=64, len=20400)
    test_data = TestData('../data/tfrecords/test/test.tfrecord', batch_size=64, len=10797)

    for idx, data in enumerate(( sorted_train_data, test_data)):
        inspect_images(data)
