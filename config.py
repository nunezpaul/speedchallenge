import argparse


class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Parameters for training model.')
        parser.add_argument('--tpu', action="store_true", default=False,
                            help='determine if to be trained on tpu')
        parser.add_argument('--opt', choices=['adam', 'sgd'], default='adam',
                            help='which optimizer to use')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='set the learning rate')
        parser.add_argument('--load_model', type=str, default=None,
                            help='file path to saved keras model to load and continue training.')
        parser.add_argument
        self.settings = vars(parser.parse_args())


if __name__ == '__main__':
    config = Config()
    for key, val in config.settings.items():
        print(key, val)