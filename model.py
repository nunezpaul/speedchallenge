import os
import uuid

import keras as k
import tensorflow as tf

from config import Config
from datasets import TrainData, ValidData
from pandas import DataFrame


class DeepVO(object):
    def __init__(self, train_data, dropout, bucket_size, load_model, opt, lr, tpu, save_dir, **kwargs):
        self.uuid = uuid.uuid4()
        self.save_dir = save_dir if save_dir else ''
        self.load_model = load_model
        self.dropout = dropout
        self.bucket_size = bucket_size
        self.num_buckets = 30 // bucket_size
        self.optimizer = self.setup_optimizer(opt, lr)
        self.callbacks = self.setup_callbacks()
        self.model = self.setup_model(train_data)

        # Convert model to tpu model
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
        tensorboard = k.callbacks.TensorBoard(log_dir=f'./log/{self.uuid}',
                                              histogram_freq=0,
                                              write_graph=True,
                                              write_images=True)
        checkpoint = k.callbacks.ModelCheckpoint(self.save_dir + '{epoch:02d}-{val_loss:.2f}.hdf5',
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
        model_input = k.layers.Input(shape=train_data.img_shape[1:])
        model_output = self.cnn(model_input)
        model = k.models.Model(inputs=model_input, outputs=model_output)

        model.compile(optimizer=self.optimizer,
                      loss=self.sparse_categorical_crossentropy,
                      metrics=[self.categorical_accuracy, self.mean_squared_error])
        model.summary()
        if self.load_model:
            print(f'Reloading pretrained {self.load_model} model.')
            model.load_weights(self.load_model)
            print(f'Successfully reloaded pretrained {self.load_model} model!')

        return model

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

        speed_prob = k.layers.Dense(self.num_buckets, activation='softmax')(fc7_dropout)

        return speed_prob

    def fit(self, epochs, train_data, valid_data=None):
        self.model.fit(train_data.img, train_data.speed,
                       epochs=epochs,
                       steps_per_epoch=train_data.len // train_data.batch_size,
                       validation_data=[valid_data.img, valid_data.speed] if valid_data else None,
                       validation_steps=valid_data.len // valid_data.batch_size if valid_data else None,
                       callbacks=self.callbacks)

    def sparse_categorical_crossentropy(self, y_speed, y_pred):
        y_true = self.bucket_speed(y_speed)
        cat_crossentropy_loss = k.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return cat_crossentropy_loss

    def mean_squared_error(self, y_speed, y_pred):
        print(y_speed, y_pred)
        y_cat = tf.argmax(y_pred, -1)
        y_cat_speed = tf.to_float((tf.to_float(y_cat) + 0.5) * self.bucket_size)
        return tf.reduce_mean(tf.square(y_cat_speed - y_speed), -1)

    def categorical_accuracy(self, y_speed, y_pred):
        y_true = self.bucket_speed(y_speed)
        y_pred = tf.argmax(y_pred, -1)
        return k.metrics.categorical_accuracy(y_true, y_pred)

    def bucket_speed(self, y_true):
        return tf.clip_by_value(y_true // self.bucket_size, 0, self.num_buckets)

    def predict(self, data, save_dir):
        filepath = (save_dir if save_dir else './') + 'prediction.txt'
        for step in range(data.len // data.batch_size + 1):
            prediction_logits = self.model.predict(data.img, steps=1)
            prediction = DataFrame(prediction_logits)
            prediction.to_csv(filepath, index=False, mode='w' if step == 0 else 'a', header=step==0)
            print(step)


if __name__ == '__main__':
    config = Config()
    if config.params['save_dir']:
        from google.colab import drive
        drive.mount('gdrive')

    train_data = TrainData('data/tfrecords/train/shard_{}.tfrecord', num_shards=10, batch_size=32, len=2000)
    valid_data = ValidData('data/tfrecords/val/val.tfrecord', batch_size=32, len=8615)

    deep_vo = DeepVO(train_data=train_data, **config.params)
    deep_vo.fit(epochs=config.params['epochs'], train_data=train_data, valid_data=valid_data)
