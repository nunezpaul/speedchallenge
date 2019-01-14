from config import Config
from datasets import TestData, TrainData
from model import DeepVO


class PredConfig(Config):
    def __init__(self):
        super(PredConfig, self).__init__()
        self.parser.add_argument('--pred_file', type=str, default=None,
                                 help='Which tfrecord to make predictions on model.')
        self.parser.add_argument('--pred_file_len', type=int, default=None,
                                 help='How many frames to predict from the tfrecord.')


if __name__ == '__main__':
    config = PredConfig()
    if config.params['save_dir']:
        from google.colab import drive
        drive.mount('gdrive')

    train_data = TrainData('data/tfrecords/train/train.tfrecord',
                           num_shards=1,
                           batch_size=32,
                           len=18360,
                           training=True,
                           class_weights_csv='model_params/class_weights.csv')
    pred_data = TestData(config.parser.pred_file, batch_size=32, len=config.parser.pred_file_len)

    deep_vo = DeepVO(train_data=train_data, **config.params)

    prediction = deep_vo.predict(pred_data, save_dir=config.params['save_dir'])