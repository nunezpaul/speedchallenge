import cv2 as cv
import csv

from random import randint
from random import shuffle


def convert_video_to_img(file):
    labels = file.format(extension='txt')
    video_file = file.format(extension='mp4')

    # read in speeds from train.txt
    with open(labels) as f:
        speeds = f.read().splitlines()

    image_label_file = 'data/image_value_pairs.csv'
    with open(image_label_file, 'w') as f:
        fieldnames = ['image_path', 'speed']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        video_capture = cv.VideoCapture(video_file)
        video_capture.set(cv.CAP_PROP_FRAME_COUNT, len(speeds))

        for idx, speed in enumerate(speeds):
            video_capture.set(cv.CAP_PROP_POS_FRAMES, idx)
            success, image = video_capture.read()

            if success:
                image_path = 'data/images/img_{idx}.jpg'.format(idx=idx)
                cv.imwrite(image_path, image)

                writer.writerow({'image_path': image_path, 'speed': speed})

            if idx % 1000 == 0:
                print(idx)

    print('done!')
    return image_label_file


def shard_data(filename, num_shards=2000):
    datafilenames = {}
    datafiles = {}
    shards = [str(i) for i in range(num_shards)]

    # Create filenames and open shard files to be written
    for shard in shards:
        datafilenames[shard] = 'data/sharded_image_value_pairs/image_value_pairs_{shard}.csv'.format(shard=shard)
        datafiles[shard] = open(datafilenames[shard], 'w')

    # Read each line after the header of training.csv
    with open(filename, 'r') as f:
        header = f.readline()
        for shard in shards:
            datafiles[shard].write(header)

        # Split the data line into training or testing
        for line in f:
            which_shard = str(randint(0, num_shards-1))
            datafiles[which_shard].write(line)

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
    image_label_file = convert_video_to_img(file)
    shard_file_names = shard_data(image_label_file)
    shuffle_sharded_data(shard_file_names)
