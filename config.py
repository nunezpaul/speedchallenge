import argparse


class Config(object):
    def __init__(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(description='Parameters for training model.')
        parser.add_argument('--tpu', action="store_true", default=False,
                            help='determine if to be trained on tpu')
        parser.add_argument('--opt', choices=['adam', 'sgd'], default='adam',
                            help='which optimizer to use')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='set the learning rate')
        parser.add_argument('--load_model', type=str, default=None,
                            help='file path to saved keras model to load.')
        parser.add_argument('--bucket_size_file', type=str, default='model_params/bucket_size.txt',
                            help='Set the bucket sizing that the data will be categorized into.')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout percent to keep during training.')
        parser.add_argument('--save_dir', type=str, default=None,
                            help='Where to save the trained model.')
        parser.add_argument('--epochs', type=int, default=100,
                            help='How many epochs to train the model for.')
        parser.add_argument('--categorical_weight', type=float, default=1,
                            help='This factor is multiplied by the categorical loss before adding to all losses.')
        parser.add_argument('--speed_weight', type=float, default=0,
                            help='This factor is multiplied by the speed loss before adding to all losses.')
        self.parser = parser
        self.params = vars(parser.parse_args())


if __name__ == '__main__':
    config = Config()
    for key, val in config.settings.items():
        print(key, val)
