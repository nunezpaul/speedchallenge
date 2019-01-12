import os
import uuid

import keras as k
import tensorflow as tf

from config import Config
from datasets import TrainData, ValidData
from pandas import DataFrame, read_csv


class DeepVO(object):
    def __init__(self, train_data, dropout, bucket_size, load_model, opt, lr, tpu, save_dir, **kwargs):
        self.uuid = uuid.uuid4()
        self.save_dir = save_dir if save_dir else './'
        self.load_model = load_model
        self.dropout = dropout
        self.bucket_size = bucket_size
        self.bucket_size_tf = tf.constant(self.bucket_size, dtype=tf.float32, shape=(1, 1))
        self.num_buckets = 30 // bucket_size
        self.optimizer = self.setup_optimizer(opt, lr)
        self.callbacks = self.setup_callbacks()
        self.model = self.setup_model(train_data)

        # Convert model to tpu model
        # TODO: Fix the model so it works with TPU
        if tpu:
            tpu_worker = 'grpc://' + os.environ['COLAB_TPU_ADDR']
            strategy = tf.contrib.tpu.TPUDistributionStrategy(
                    tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_worker))
            self.model = tf.contrib.tpu.keras_to_tpu_model(self.model, strategy=strategy)

    def setup_optimizer(self, opt, lr):
        if self.load_model:
            print('Reloading previous optimizer state.')
            prev_model = k.models.load_model(self.load_model, custom_objects={'tf': tf, 'k': k})
            opt = prev_model.optimizer
            print('Complete!')
        elif opt == 'sgd':
            opt = k.optimizers.SGD(lr=lr, momentum=0.9)
        else:
            opt = k.optimizers.adam(lr=lr)
        return opt

    def setup_callbacks(self):
        callbacks = []
        tensorboard_dir = f'{self.save_dir}log/'
        saved_model_dir = f'{self.save_dir}models/{self.uuid}/'

        for directory in (tensorboard_dir, saved_model_dir):
            if not os.path.exists(directory):
                os.makedirs(directory)

        tensorboard = k.callbacks.TensorBoard(log_dir=f'{self.save_dir}log/{self.uuid}',
                                              histogram_freq=0,
                                              write_graph=True,
                                              write_images=True)
        model_name = '{epoch:02d}_{val_loss:.2f}.hdf5'
        checkpoint = k.callbacks.ModelCheckpoint(saved_model_dir + model_name,
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='auto',
                                                 period=1)
        reduce_lr = k.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.2,
                                                  patience=5,
                                                  min_lr=10 ** -7)
        callbacks += [tensorboard, checkpoint, reduce_lr]
        return callbacks

    def setup_model(self, train_data):
        print(train_data.speed.shape)
        model_input = k.layers.Input(shape=train_data.img_shape[1:])
        model_output = self.cnn(model_input)
        model = k.models.Model(inputs=model_input, outputs=model_output)

        losses = {'category': k.losses.sparse_categorical_crossentropy, 'speed': self.mean_squared_error}
        loss_weights = {'category': 1.0, 'speed': 0.0}
        metrics = {'category': k.metrics.sparse_categorical_accuracy}

        model.compile(optimizer=self.optimizer,
                      loss=losses,
                      loss_weights=loss_weights,
                      metrics=metrics)
        model.summary()
        if self.load_model:
            print(f'Reloading pretrained {self.load_model} model.')
            model.load_weights(self.load_model)
            print(f'Successfully reloaded pretrained {self.load_model} model!')

        return model

    def sparse_categorical_crossentropy(self, y_true, y_pred):
        cat_crossentropy_loss = k.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return cat_crossentropy_loss

    def mean_squared_error(self, y_true, y_pred):
        speed = self.convert_to_speed(y_pred)
        return k.losses.mean_squared_error(y_true, speed)

    def convert_to_speed(self, category_output):
        category = tf.to_float(tf.argmax(category_output, axis=-1)) + 0.5
        speed_output = tf.multiply(category, self.bucket_size)
        return speed_output

    def cnn(self, model_input):
        conv1 = k.layers.ZeroPadding2D((3, 3))(model_input)
        conv1 = k.layers.Conv2D(64, kernel_size=(7, 7), strides=2, padding='valid', name='conv1')(conv1)
        conv1 = k.layers.BatchNormalization()(conv1)
        conv1 = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu1')(conv1)

        conv2 = k.layers.ZeroPadding2D((2, 2))(conv1)
        conv2 = k.layers.Conv2D(128, kernel_size=(5, 5), strides=2, padding='valid', name='conv2')(conv2)
        conv2 = k.layers.BatchNormalization()(conv2)
        conv2 = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu2')(conv2)

        conv3a = k.layers.ZeroPadding2D((2, 2))(conv2)
        conv3a = k.layers.Conv2D(256, kernel_size=(5, 5), strides=2, padding='valid', name='conv3a')(conv3a)
        conv3a = k.layers.BatchNormalization()(conv3a)
        conv3a = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu3a')(conv3a)

        conv3b = k.layers.ZeroPadding2D()(conv3a)
        conv3b = k.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='valid', name='conv3b')(conv3b)
        conv3b = k.layers.BatchNormalization()(conv3b)
        conv3b = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu3b')(conv3b)

        conv4 = k.layers.ZeroPadding2D()(conv3b)
        conv4 = k.layers.Conv2D(512, kernel_size=(3, 3), strides=2, padding='valid', name='conv4a')(conv4)
        conv4 = k.layers.BatchNormalization()(conv4)
        conv4 = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu4')(conv4)

        conv4b = k.layers.ZeroPadding2D()(conv4)
        conv4b = k.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='valid', name='conv4b')(conv4b)
        conv4b = k.layers.BatchNormalization()(conv4b)
        conv4b = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu4b')(conv4b)

        conv5a = k.layers.ZeroPadding2D()(conv4b)
        conv5a = k.layers.Conv2D(512, kernel_size=(3, 3), strides=2, padding='valid', name='conv5a')(conv5a)
        conv5a = k.layers.BatchNormalization()(conv5a)
        conv5a = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu5a')(conv5a)

        conv5b = k.layers.ZeroPadding2D()(conv5a)
        conv5b = k.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='valid', name='conv5b')(conv5b)
        conv5b = k.layers.BatchNormalization()(conv5b)
        conv5b = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu5b')(conv5b)

        conv6 = k.layers.ZeroPadding2D()(conv5b)
        conv6 = k.layers.Conv2D(1024, kernel_size=(3, 3), strides=2, padding='valid', name='conv6')(conv6)
        conv6 = k.layers.BatchNormalization()(conv6)
        conv6 = k.layers.MaxPool2D(pool_size=(2, 3))(conv6)  # only difference in addition from original paper

        flat = k.layers.Flatten()(conv6)

        fc6 = k.layers.Dense(1024, activation='relu', name='fc6')(flat)
        fc6_dropout = k.layers.Dropout(self.dropout)(fc6)

        fc7 = k.layers.Dense(1024, activation='relu', name='fc7')(fc6_dropout)
        fc7_dropout = k.layers.Dropout(self.dropout)(fc7)

        category_output = k.layers.Dense(self.num_buckets, activation='softmax', name='category')(fc7_dropout)
        speed_output = k.layers.Lambda(lambda x: x, name='speed')(category_output)

        return category_output, speed_output

    def fit(self, epochs, train_data, valid_data=None):
        if valid_data:
            validation_data = [valid_data.img, {'category': valid_data.label, 'speed': valid_data.speed}]
        self.model.fit(train_data.img, {'category': train_data.label, 'speed': train_data.speed},
                       class_weight=train_data.class_weights,
                       epochs=epochs,
                       steps_per_epoch=train_data.len // train_data.batch_size,
                       validation_data=validation_data if valid_data else None,
                       validation_steps=valid_data.len // valid_data.batch_size if valid_data else None,
                       callbacks=self.callbacks)

    def predict(self, data, save_dir):
        filepath = (save_dir if save_dir else './') + 'prediction.txt'
        for step in range(data.len // data.batch_size + 1):
            prediction_logits = self.model.predict(data.img, steps=1)
            prediction = DataFrame(prediction_logits)
            prediction.to_csv(filepath, index=False, mode='w' if step == 0 else 'a', header=step==0)


if __name__ == '__main__':
    config = Config()
    if config.params['save_dir']:
        from google.colab import drive
        drive.mount('gdrive', force_remount=False)

    valid_data = ValidData('data/tfrecords/val/val.tfrecord', batch_size=32, len=2040)
    train_data = TrainData('data/tfrecords/train/train.tfrecord',
                           num_shards=1,
                           batch_size=32,
                           len=18360,
                           training=True,
                           class_weights_csv='data/labeled_csv/train/train_class_weights.csv')

    deep_vo = DeepVO(train_data=train_data, **config.params)
    deep_vo.fit(epochs=config.params['epochs'], train_data=train_data, valid_data=valid_data)
