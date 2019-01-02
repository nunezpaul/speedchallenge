import argparse
import os

import pandas as pd


def get_img_pairs(test_img_dir):
    num_imgs = len(os.listdir(test_img_dir))
    prev_img = []
    curr_img = []
    for i in range(num_imgs):
        prev_img.append(f'data/images/test/img{i}.jpg')
        curr_img.append(f'data/images/test/img{i + 1}.jpg')

    img_pairs = pd.DataFrame()
    img_pairs['prev_img'] = prev_img
    img_pairs['curr_img'] = curr_img

    return img_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for creating the data pairs.')
    parser.add_argument('--test_img_dir', type=str, default='data/images/test',
                        help='file path for where the test images are listed.')
    params = vars(parser.parse_args())

    img_pairs = get_img_pairs(params['test_img_dir'])
    img_pairs.to_csv('data/labeled_csv/test/test.csv', index=False)