import csv

from random import shuffle


def write_image_value_pairs(file):
    labels = file.format(extension='txt')

    # read in speeds from train.txt
    with open(labels) as f:
        speeds = f.read().splitlines()

    image_label_file = 'data/image_value_pairs.csv'
    with open(image_label_file, 'w') as f:
        fieldnames = ['prev_image_path', 'curr_image_path', 'speed']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for idx, speed in enumerate(speeds):
            if idx % 2 == 0:
                continue

            curr_image_path = 'data/images/img{idx}.jpg'.format(idx=idx)
            prev_image_path = 'data/images/img{idx}.jpg'.format(idx=idx - 1)
            writer.writerow({'prev_image_path': prev_image_path, 'curr_image_path': curr_image_path, 'speed': speed})

    print('done!')
    return image_label_file


def shard_data(filename, num_shards=10):
    datafilenames = {}
    datafiles = {}
    shards = [i for i in range(num_shards)]

    # Create filenames and open shard files to be written
    for shard in shards:
        datafilenames[shard] = 'data/sharded_image_value_pairs/image_value_pairs_{shard}.csv'.format(shard=shard)
        datafiles[shard] = open(datafilenames[shard], 'w')

    # Read each line after the header of training.csv
    with open(filename, 'r') as f:
        data = f.read().splitlines()
        shuffle(data)

    num_data_per_shard = int(len(data)/num_shards + 1)

    for idx, line in enumerate(data):
        which_shard = int(idx / num_data_per_shard)
        datafiles[which_shard].write(line + '\n')

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
    file = 'data/train.{extension}'
    image_label_file = write_image_value_pairs(file)
    shard_file_names = shard_data(image_label_file)
    shuffle_sharded_data(shard_file_names)