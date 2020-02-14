import os

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Activation, Dropout, RepeatVector, TimeDistributed
from keras_tqdm import TQDMCallback
from numpy import clip, inf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def common_plot_prediction(ground_truth, prediction, term_name, model_name, look_back, look_ahead, hidden_neurons,
                           num_epochs):
    fig = plt.figure(term_name, figsize=(20, 4), dpi=100)
    fig.suptitle(f'Model {model_name} with {hidden_neurons} hidden neurons '
                 f'and {num_epochs} epochs\n'
                 f'Prediction of {term_name} of {len(prediction)} samples ahead, '
                 f'trained on {look_back} look back', fontsize=16)
    ax = fig.add_subplot(111)
    ax.plot(ground_truth, color='b', linestyle='-', marker='x', label='Ground truth')
    ax.plot(range(len(ground_truth) - len(prediction), len(ground_truth)), prediction,
            color='r', linestyle='-', marker='+', label='Prediction')
    ax.set_xlabel(term_name, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    folder_name = os.path.join('lstm-results', model_name)
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(
        os.path.join(folder_name, f'{term_name}-hiddenunits={hidden_neurons}-back={look_back}-ahead={look_ahead}.png'))
    plt.close()


class LstmForecasterMultipleLookAhead(object):

    def __create_sliding_window_dataset(self, input_time_series, look_back, look_forward):
        data_x, data_y = [], []
        for index in range(len(input_time_series) - look_back - look_forward):
            data_x.append(input_time_series[index:index + look_back])
            data_y.append(input_time_series[index + look_back: index + look_back + look_forward])

        return np.array(data_x), np.array(data_y)

    def __init__(self, time_series, look_back, look_forward, term_name, stateful=True, random_seed=None, verbose=False):

        self.__verbose = verbose
        self.__term_name = term_name

        if random_seed is not None:
            np.random.seed(random_seed)

        self.__source_time_series = np.reshape(time_series.astype(dtype=np.float), newshape=(-1, 1))
        self.__look_back = look_back
        self.__look_forward = look_forward

        time_series_delta = time_series[1:] - time_series[:-1]
        self.__source_time_series_delta = np.reshape(time_series_delta.astype(dtype=np.float), newshape=(-1, 1))

        self.__scaler = MinMaxScaler(feature_range=(-1, 1))
        data_set = self.__scaler.fit_transform(self.__source_time_series_delta)
        data_set_x, data_set_y = self.__create_sliding_window_dataset(data_set, look_back, look_forward)
        self.__test_train_split = 0.67
        train_size = int(len(data_set_x) * self.__test_train_split)
        train_x = data_set_x[0:train_size, :]
        train_y = data_set_y[0:train_size, :]

        test_x = data_set_x[train_size:, :]
        test_y = data_set_y[train_size:, :]

        self.__hidden_neurons = 100  # 20  # 100

        if stateful:
            self.__model_name = 'encode-decode-stateful'
            self.__num_epochs = 200
            batch_size = 1

            self.__model = Sequential()
            self.__model.add(
                LSTM(batch_input_shape=(batch_size, None, 1), units=self.__hidden_neurons, return_sequences=False,
                     stateful=True))
            self.__model.add(RepeatVector(look_forward))
            self.__model.add(LSTM(units=self.__hidden_neurons, return_sequences=True, stateful=True))
            self.__model.add(TimeDistributed(Dense(1)))
            self.__model.add(Activation('linear'))
            if self.__verbose:
                self.__model.summary(print_fn=print)
            self.__model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

            epoch_iterator = range(self.__num_epochs)
            if self.__verbose:
                epoch_iterator = tqdm(epoch_iterator, unit='epoch')
            for _ in epoch_iterator:
                history = self.__model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=1,
                                           batch_size=batch_size, verbose=0)
                val_loss = history.history['val_loss'][0]
                loss = history.history['loss'][0]
                if self.__verbose:
                    epoch_iterator.set_description(f"val loss={val_loss:0.3f} loss={loss:0.3f}")
                self.__model.reset_states()
        else:
            self.__model_name = 'encode-decode-stateless'
            self.__model = Sequential()
            self.__model.add(LSTM(input_shape=(None, 1), units=self.__hidden_neurons, return_sequences=False))
            self.__model.add(RepeatVector(look_forward))
            self.__model.add(LSTM(units=self.__hidden_neurons, return_sequences=True))
            self.__model.add(TimeDistributed(Dense(1)))
            self.__model.add(Activation('linear'))
            if self.__verbose:
                self.__model.summary(print_fn=print)
            self.__model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='loss', min_delta=0.000001,
                                           patience=50)  # this big patience is important
            callbacks = [early_stopping]
            if self.__verbose:
                callbacks.append(TQDMCallback())
            history = self.__model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=10000,
                                       callbacks=callbacks, verbose=0)
            self.__num_epochs = len(history.epoch)

    @property
    def configuration(self):
        return f'TTSplit={self.__test_train_split:.2} numEpochs={self.__num_epochs} numHN={self.__hidden_neurons}'

    def predict_counts(self, time_series=None):
        if time_series is None:
            time_series = self.__source_time_series
            source_time_series_delta = self.__source_time_series_delta
        else:
            time_series_delta = time_series[1:] - time_series[:-1]
            source_time_series_delta = np.reshape(time_series_delta.astype(dtype=np.float), newshape=(-1, 1))

        data_set = self.__scaler.fit_transform(source_time_series_delta.astype(dtype=np.float))

        data_set_seed = data_set[-self.__look_back:]
        reshaped_seed = np.reshape(data_set_seed, (1, -1, 1))
        predicted_sequence_scaled_diff = self.__model.predict(reshaped_seed).reshape(-1, 1)
        predicted_sequence_diff = self.__scaler.inverse_transform(predicted_sequence_scaled_diff)

        previous_predicted_value = time_series[-1]
        predicted_values = []
        for step in range(self.__look_forward):
            predicted_scaled_diff = previous_predicted_value + predicted_sequence_diff[step][0]
            predicted_values.append(predicted_scaled_diff)
            previous_predicted_value = predicted_scaled_diff

        return clip(predicted_values, 0, inf)

    def plot_prediction(self, ground_truth, prediction):
        common_plot_prediction(ground_truth, prediction, self.__term_name, self.__model_name, self.__look_back,
                               self.__look_forward, self.__hidden_neurons, self.__num_epochs)


class LstmForecasterSingleLookAhead(object):

    @staticmethod
    def __create_sliding_window_dataset(input_time_series, look_back):
        data_x, data_y = [], []
        for index in range(len(input_time_series) - look_back - 1):
            single_window = input_time_series[index:(index + look_back)]
            data_x.append(single_window)
            data_y.append(input_time_series[index + look_back])

        return np.array(data_x), np.array(data_y)

    def __init__(self, time_series, look_back, look_forward, term_name, stateful=True, random_seed=None, verbose=False):

        self.__verbose = verbose
        self.__term_name = term_name

        if random_seed is not None:
            np.random.seed(random_seed)

        self.__source_time_series = np.reshape(time_series.astype(dtype=np.float), newshape=(-1, 1))
        self.__look_back = look_back
        self.__look_forward = look_forward

        time_series_delta = time_series[1:] - time_series[:-1]
        self.__source_time_series_delta = np.reshape(time_series_delta.astype(dtype=np.float), newshape=(-1, 1))

        self.__scaler = MinMaxScaler(feature_range=(-1, 1))
        data_set = self.__scaler.fit_transform(self.__source_time_series_delta)
        data_set_x, data_set_y = self.__create_sliding_window_dataset(data_set, look_back)
        self.__test_train_split = 0.67
        train_size = int(len(data_set_x) * self.__test_train_split)
        train_x = data_set_x[0:train_size, :]
        train_y = data_set_y[0:train_size, :]

        test_x = data_set_x[train_size:, :]
        test_y = data_set_y[train_size:, :]

        self.__hidden_neurons = 100  # 20  # 100

        if stateful:
            self.__model_name = 'single-look-ahead-stateful'
            self.__num_epochs = 40
            batch_size = 1

            self.__model = Sequential()
            self.__model.add(LSTM(batch_input_shape=(batch_size, None, 1), units=self.__hidden_neurons,
                                  input_shape=(self.__look_back, 1), return_sequences=False, stateful=True))
            self.__model.add(Dropout(0.2))
            self.__model.add(Dense(1))
            self.__model.add(Activation('linear'))
            if self.__verbose:
                self.__model.summary(print_fn=print)
            self.__model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

            epoch_iterator = range(self.__num_epochs)
            if self.__verbose:
                epoch_iterator = tqdm(epoch_iterator, unit='epoch')
            for _ in epoch_iterator:
                history = self.__model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=1,
                                           batch_size=batch_size, verbose=0)
                val_loss = history.history['val_loss'][0]
                loss = history.history['loss'][0]
                if self.__verbose:
                    epoch_iterator.set_description(f"val loss={val_loss:0.3f} loss={loss:0.3f}")
                self.__model.reset_states()

        else:
            self.__model_name = 'single-look-ahead-stateless'
            self.__num_epochs = 200

            self.__model = Sequential()
            self.__model.add(
                LSTM(units=self.__hidden_neurons, input_shape=(self.__look_back, 1), return_sequences=False,
                     stateful=False))
            self.__model.add(Dropout(0.2))
            self.__model.add(Dense(1))
            self.__model.add(Activation('linear'))
            if self.__verbose:
                self.__model.summary(print_fn=print)
            self.__model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

            self.__history = self.__model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y),
                                              epochs=self.__num_epochs, verbose=0)

    @property
    def configuration(self):
        return f'TTSplit={self.__test_train_split:.2} numEpochs={self.__num_epochs} numHN={self.__hidden_neurons}'

    def predict_counts(self, time_series=None):
        if time_series is None:
            time_series = self.__source_time_series
            source_time_series_delta = self.__source_time_series_delta
        else:
            time_series_delta = time_series[1:] - time_series[:-1]
            source_time_series_delta = np.reshape(time_series_delta.astype(dtype=np.float), newshape=(-1, 1))

        data_set = self.__scaler.fit_transform(source_time_series_delta.astype(dtype=np.float))

        data_set_seed = data_set[-self.__look_back:]
        predicted_sequence_scaled_diff = []
        for step in range(self.__look_forward):
            reshaped_seed = np.reshape(data_set_seed, (1, -1, 1))
            predicted_scaled_diff = self.__model.predict(reshaped_seed)
            predicted_sequence_scaled_diff.append(predicted_scaled_diff[0])
            data_set_seed = np.roll(data_set_seed, -1)
            data_set_seed[-1] = predicted_scaled_diff

        predicted_sequence_diff = self.__scaler.inverse_transform(predicted_sequence_scaled_diff)

        previous_predicted_value = time_series[-1]
        predicted_values = []
        for step in range(self.__look_forward):
            predicted_scaled_diff = previous_predicted_value + predicted_sequence_diff[step][0]
            predicted_values.append(predicted_scaled_diff)
            previous_predicted_value = predicted_scaled_diff

        return clip(predicted_values, 0, inf)

    def plot_prediction(self, ground_truth, prediction):
        common_plot_prediction(ground_truth, prediction, self.__term_name, self.__model_name, self.__look_back,
                               self.__look_forward, self.__hidden_neurons, self.__num_epochs)


class LstmForecasterMultipleModelSingleLookAhead(object):

    @staticmethod
    def __create_sliding_window_dataset(input_time_series, look_back, look_ahead_offset):
        data_x, data_y = [], []
        for index in range(len(input_time_series) - look_back - look_ahead_offset):
            single_window = input_time_series[index:(index + look_back)]
            data_x.append(single_window)
            data_y.append(input_time_series[index + look_back + look_ahead_offset])

        return np.array(data_x), np.array(data_y)

    def __init__(self, time_series, look_back, look_forward, term_name, stateful=True, random_seed=None, verbose=False):

        self.__verbose = verbose
        self.__term_name = term_name

        if random_seed is not None:
            np.random.seed(random_seed)

        self.__source_time_series = np.reshape(time_series.astype(dtype=np.float), newshape=(-1, 1))
        self.__look_back = look_back
        self.__look_forward = look_forward

        time_series_delta = time_series[1:] - time_series[:-1]
        self.__source_time_series_delta = np.reshape(time_series_delta.astype(dtype=np.float), newshape=(-1, 1))

        self.__scaler = MinMaxScaler(feature_range=(-1, 1))
        data_set = self.__scaler.fit_transform(self.__source_time_series_delta)

        self.__models = [None] * look_forward

        for look_forward_index in range(look_forward):
            data_set_x, data_set_y = self.__create_sliding_window_dataset(data_set, look_back, look_forward_index)
            self.__test_train_split = 0.67
            train_size = int(len(data_set_x) * self.__test_train_split)
            train_x = data_set_x[0:train_size, :]
            train_y = data_set_y[0:train_size, :]

            test_x = data_set_x[train_size:, :]
            test_y = data_set_y[train_size:, :]

            self.__hidden_neurons = 100  # 20  # 100

            if stateful:
                self.__model_name = 'single-look-ahead-multiple-model-stateful'
                self.__num_epochs = 40
                batch_size = 1

                self.__models[look_forward_index] = Sequential()
                self.__models[look_forward_index].add(
                    LSTM(batch_input_shape=(batch_size, None, 1), units=self.__hidden_neurons,
                         input_shape=(self.__look_back, 1), return_sequences=False, stateful=True))
                self.__models[look_forward_index].add(Dropout(0.2))
                self.__models[look_forward_index].add(Dense(1))
                self.__models[look_forward_index].add(Activation('linear'))
                if self.__verbose:
                    self.__models[look_forward_index].summary(print_fn=print)
                self.__models[look_forward_index].compile(loss='mean_squared_error', optimizer='rmsprop',
                                                          metrics=['accuracy'])

                epoch_iterator = range(self.__num_epochs)
                if self.__verbose:
                    epoch_iterator = tqdm(epoch_iterator, unit='epoch')
                for _ in epoch_iterator:
                    history = self.__models[look_forward_index].fit(x=train_x, y=train_y,
                                                                    validation_data=(test_x, test_y), epochs=1,
                                                                    batch_size=batch_size, verbose=0)
                    val_loss = history.history['val_loss'][0]
                    loss = history.history['loss'][0]
                    if self.__verbose:
                        epoch_iterator.set_description(f"val loss={val_loss:0.3f} loss={loss:0.3f}")
                    self.__models[look_forward_index].reset_states()

            else:
                self.__model_name = 'single-look-ahead-multiple-model-stateless'
                self.__num_epochs = 200

                self.__models[look_forward_index] = Sequential()
                self.__models[look_forward_index].add(
                    LSTM(units=self.__hidden_neurons, input_shape=(self.__look_back, 1), return_sequences=False,
                         stateful=False))
                self.__models[look_forward_index].add(Dropout(0.2))
                self.__models[look_forward_index].add(Dense(1))
                self.__models[look_forward_index].add(Activation('linear'))
                if self.__verbose:
                    self.__models[look_forward_index].summary(print_fn=print)
                self.__models[look_forward_index].compile(loss='mean_squared_error', optimizer='rmsprop',
                                                          metrics=['accuracy'])

                self.__history = self.__models[look_forward_index].fit(x=train_x, y=train_y,
                                                                       validation_data=(test_x, test_y),
                                                                       epochs=self.__num_epochs, verbose=0)

    @property
    def configuration(self):
        return f'TTSplit={self.__test_train_split:.2} numEpochs={self.__num_epochs} numHN={self.__hidden_neurons}'

    def predict_counts(self, time_series=None):
        if time_series is None:
            time_series = self.__source_time_series
            source_time_series_delta = self.__source_time_series_delta
        else:
            time_series_delta = time_series[1:] - time_series[:-1]
            source_time_series_delta = np.reshape(time_series_delta.astype(dtype=np.float), newshape=(-1, 1))

        data_set = self.__scaler.fit_transform(source_time_series_delta.astype(dtype=np.float))

        data_set_seed = data_set[-self.__look_back:]
        reshaped_seed = np.reshape(data_set_seed, (1, -1, 1))
        predicted_sequence_scaled_diff = []
        for look_ahead_index in range(self.__look_forward):
            predicted_scaled_diff = self.__models[look_ahead_index].predict(reshaped_seed)
            predicted_sequence_scaled_diff.append(predicted_scaled_diff[0])

        predicted_sequence_diff = self.__scaler.inverse_transform(predicted_sequence_scaled_diff)

        previous_predicted_value = time_series[-1]
        predicted_values = []
        for step in range(self.__look_forward):
            predicted_scaled_diff = previous_predicted_value + predicted_sequence_diff[step][0]
            predicted_values.append(predicted_scaled_diff)
            previous_predicted_value = predicted_scaled_diff

        return clip(predicted_values, 0, inf)

    def plot_prediction(self, ground_truth, prediction):
        common_plot_prediction(ground_truth, prediction, self.__term_name, self.__model_name, self.__look_back,
                               self.__look_forward, self.__hidden_neurons, self.__num_epochs)
