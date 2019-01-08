import argparse

import pandas as pd


def create_image_value_pairs(speed_file, bucket_size):
    # read in speeds from file.txt
    img_dir = speed_file.replace('/', '/images/').replace('.txt', '/img{}.jpg')
    data = pd.read_csv(speed_file, header=None, names=['speed'])
    data['category'] = data['speed'].apply(lambda x: x // bucket_size).astype(int)
    data['prev_img'] = data.index.to_series().apply(lambda x: img_dir.format(x))
    data['curr_img'] = data.index.to_series().apply(lambda x: img_dir.format(x + 1))
    data = data.iloc[:-1, :]
    data = data[['prev_img', 'curr_img', 'category', 'speed']]

    return data


def data_split(data, filename_out, split, lookback):
    # Grab every ith data point as well as the k behind it and send it to a csv file as our validation set
    for i in range(lookback):
        val_data = data.iloc[::split, :] if i == 0 else val_data.append(data.iloc[::split-i, :])
    val_data.to_csv(filename_out.replace('train', 'val'), index=False)

    # Remove every ith data point and return the resulting dataframe
    return data.drop(val_data.index)


def write_labeled_csv_data(data, filename_out, sharding, shuffle,
                           write_class_weights=False, num_shards=10, records_per_category=200):

    if not sharding:
        _write_csv(data, filename=filename_out, shuffle=shuffle, write_class_weights=write_class_weights)
        return

    filename_out = filename_out.replace('.', '_shard_{}.')
    shard_filenames = {}

    # Split the data by its respective categorical designation
    data_splits = []
    for i in range(data.category.max() + 1):
        data_splits.append(data[data.category == i])

    # TODO: separate out the sharding and sampling methods
    # Create filenames and open shard files to be written
    for shard in range(num_shards):
        shard_filenames[shard] = filename_out.format(shard)
        for idx, data_split in enumerate(data_splits):
            with_replacement = data_split.shape[0] < records_per_category
            sample = data_split.sample(records_per_category, replace=with_replacement)
            if idx == 0:
                data_to_write = sample
            else:
                data_to_write = data_to_write.append(sample)

        _write_csv(data_to_write, shard_filenames[shard], shuffle=shuffle)

    return shard_filenames


def _write_csv(dataframe, filename, shuffle, write_class_weights):
    if write_class_weights:
        class_weights = (dataframe.groupby('category').nunique()['curr_img'] / dataframe.shape[0]) ** -1
        class_weights.to_csv(filename.replace('.', '_class_weights.'), index=False)
    if shuffle:
        dataframe = dataframe.sample(frac=1.0).reset_index(drop=True)
    dataframe.to_csv(filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for creating the data pairs.')
    parser.add_argument('--speed_file', type=str, default='data/train.txt',
                        help='file path for where the speeds are listed.')
    parser.add_argument('--output_file', type=str, default='data/labeled_csv/train/train_shard.csv',
                        help='file path for where the labeled data is going to be written to.')
    parser.add_argument('--bucket_size', type=int, default=3,
                        help='Size the grouping window for all speeds')
    parser.add_argument('--split_inc', type=int, default=10,
                        help='Size in which every ith data point is sent as a validation point.')
    parser.add_argument('--lookback', type=int, default=5,
                        help='If splitting data in train/val set then this is how far back the separation will start. '
                             'e.g. a split of 20 and look back of 5 means that every 16th, 17th, 18th, 19th and 20th '
                             'will be reserved as validation data.')
    parser.add_argument('--data_split', action="store_true", default=False,
                        help='Takes every 10th data point and sends it to be validation data.')
    parser.add_argument('--shard', action="store_true", default=False,
                        help='Takes every 10th data point and sends it to be validation data.')
    parser.add_argument('--shuffle', action="store_true", default=False,
                        help='Shuffle the data before writing the data.')
    parser.add_argument('--write_class_weights', action="store_true", default=False,
                        help='Write how much each class should be weight. Based on inverse percentage of dataset.')
    parser.add_argument('--records_per_category', type=int, default=200,
                        help='How many samples from each category will be taken.')
    parser.add_argument('--num_shards', type=int, default=10,
                        help='How many shards will be created from the data to be written.')
    params = vars(parser.parse_args())

    file_in = params['speed_file']
    data = create_image_value_pairs(file_in, bucket_size=params['bucket_size'])
    if params['data_split']:
        data = data_split(data,
                          lookback=params['lookback'],
                          filename_out=params['output_file'],
                          split=params['split_inc'])
    shard_filenames = write_labeled_csv_data(data, params['output_file'],
                                             sharding=params['shard'],
                                             records_per_category=params['records_per_category'],
                                             shuffle=params['shuffle'],
                                             write_class_weights=params['write_class_weights'])