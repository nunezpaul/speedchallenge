import csv
import os

import pandas as pd

from random import shuffle


def create_speed_category_file(file, bucket_size=3):
    data = pd.read_csv(file, header=None)
    data.columns = ['Speed']
    data //= bucket_size
    data = data.astype(int)

    num_cats = data.max().values[0] + 1
    file_out = file.replace('train', f'train_cat_{num_cats}')
    data.to_csv(file_out, index=False)
    return file_out, num_cats


def write_image_value_pairs(file, look_back=2):
    # read in speeds from train_cat.txt
    with open(file) as f:
        speeds = f.read().splitlines()

    image_label_file = 'data/image_value_pairs.csv'
    with open(image_label_file, 'w') as f:
        fieldnames = list(range(look_back)) + ['speed']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for idx, speed in enumerate(speeds):
            if idx < (look_back - 1):
                continue

            write_dict = {'speed': speed}
            for i in range(look_back):
                write_dict[i] = 'data/train_images/img{idx}.jpg'.format(idx=idx - i)
            writer.writerow(write_dict)

    print('done!')
    return image_label_file


def train_test_split(filename, split=10):
    datafilenames = {}
    datafiles = {}

    # Create filenames and open train/test files to be written
    for data_type in ('train', 'val'):
        datafilenames[data_type] = 'data/{type}_image_value_pairs.csv'.format(type=data_type)
        datafiles[data_type] = open(datafilenames[data_type], 'w')

    # Read each line of the main image_value_pair
    with open(filename) as f:
        data = f.read().splitlines()

    # Separating the files now. Every Kth data point is for testing
    count = 1
    for idx, line in enumerate(data):
        if count == split:
            count = 1
            datafiles['val'].write(line + '\n')
            continue

        datafiles['train'].write(line + '\n')
        count += 1

    print('done!')
    return datafilenames['train']


def shard_data(filename, num_cats, num_shards=10, records_per_category=200):
    datafilenames = {}
    datafiles = {}

    # Split the data by its respective categorical designation
    data = pd.read_csv(filename, header=None)
    cat_idx = data.shape[-1] - 1
    data_splits = []
    for i in range(num_cats):
        data_splits.append(data[data[cat_idx] == i])

    # Create filenames and open shard files to be written
    for shard in range(num_shards):
        datafilenames[shard] = 'data/sharded_image_value_pairs/image_value_pairs_{shard}.csv'.format(shard=shard)
        with open(datafilenames[shard], 'w') as f:
            for data_split in data_splits:
                need_replacement = data_split.shape[0] < records_per_category
                sample = data_split.sample(records_per_category, replace=need_replacement)
                sample.to_csv(f, header=False, index=False)

    return datafilenames


def shuffle_sharded_data(filenames):
    for key, filename in filenames.items():
        with open(filename) as f:
            datalines = f.readlines()
            shuffle(datalines)

        with open(filename, 'w') as f:
            for dataline in datalines:
                f.writelines(dataline)
    print('done!')


if __name__ == '__main__':
    file = 'data/train.txt'
    cat_file, num_cats = create_speed_category_file(file)
    image_label_file = write_image_value_pairs(cat_file)
    train_image_label_file = train_test_split(image_label_file)
    shard_file_names = shard_data(train_image_label_file, num_cats)
    shuffle_sharded_data(shard_file_names)

    # remove the intermediate files 
    os.remove('data/image_value_pairs.csv')
    os.remove('data/train_image_value_pairs.csv')