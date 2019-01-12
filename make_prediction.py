from config import Config
from datasets import TestData, TrainData, ValidData
from model import DeepVO


if __name__ == '__main__':
    config = Config()
    if config.params['save_dir']:
        from google.colab import drive
        drive.mount('gdrive')

    train_data = TrainData('data/tfrecords/train/train.tfrecord',
                           num_shards=1,
                           batch_size=32,
                           len=18360,
                           training=True,
                           class_weights_csv='data/labeled_csv/train/train_class_weights.csv')
    valid_data = ValidData('data/tfrecords/val/sorted_train.tfrecord', batch_size=32, len=20400)

    deep_vo = DeepVO(train_data=train_data, **config.params)

    prediction = deep_vo.predict(valid_data, save_dir=config.params['save_dir'])