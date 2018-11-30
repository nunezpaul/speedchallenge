# import tensorflow as tf
import cv2 as cv
import os

from random import randint

# tf.enable_eager_execution()
# tf.executing_eagerly()

def basic_model():
    pass


def generate_which_shard():
    print(os.getcwd())
    shard = randint(0, 2000)
    shard_file = 'data/sharded_image_value_pairs/image_value_pairs_{}.csv'.format(shard)
    with open(shard_file) as f:
        pairs = f.read().splitlines()

    print(pairs)

if __name__ == '__main__':
    print(generate_which_shard())