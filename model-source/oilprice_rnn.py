import keras
import json
import os.path
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop
import os
from os import environ
from keras.callbacks import TensorBoard
from emetrics import EMetrics
import pandas as pd
from math import sqrt
import numpy as np
# from sklearn.preprocessing import MinMaxScaler



model_filename = "oilprice_rnn.h5"
data_filename = "WCOILWTICO.csv"



def getCurrentSubID():
    if "SUBID" in os.environ:
        return os.environ["SUBID"]
    else:
        return None

def gen_datasets(dataset, prev_periods=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - prev_periods):
        a = dataset[i:(i + prev_periods), 0]
        dataX.append(a)
        dataY.append(dataset[i + prev_periods, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

class HPOMetrics(keras.callbacks.Callback):
    def __init__(self):
        self.emetrics = EMetrics.open(getCurrentSubID())

    def on_epoch_end(self, epoch, logs={}):
        train_results = {}
        test_results = {}

        for key, value in logs.items():
            if 'val_' in key:
                test_results.update({key: value})
            else:
                train_results.update({key: value})

        print('EPOCH ' + str(epoch))
        self.emetrics.record("train", epoch, train_results)
        self.emetrics.record(EMetrics.TEST_GROUP, epoch, test_results)

    def close(self):
        self.emetrics.close()

###############################################################################

if __name__ == '__main__':

    ###############################################################################
    # Set up working directories for data, model and logs.
    ###############################################################################
    # writing the train model and getting input data
    if environ.get('DATA_DIR') is not None:
        input_data_folder = environ.get('DATA_DIR')
        input_data_path = os.path.join(input_data_folder, data_filename)
    else:
         input_data_path= data_filename

    if environ.get('RESULT_DIR') is not None:
        output_model_folder = os.path.join(os.environ["RESULT_DIR"], "model")
        output_model_path = os.path.join(output_model_folder, model_filename)
    else:
        output_model_folder = "model"
        output_model_path = os.path.join("model", model_filename)

    os.makedirs(output_model_folder, exist_ok=True)

    #writing metrics
    if environ.get('JOB_STATE_DIR') is not None:
        tb_directory = os.path.join(os.environ["SUBID"], os.environ["JOB_STATE_DIR"], "logs", "tb", "test")
    else:
        tb_directory = os.path.join("logs", "tb", "test")

    os.makedirs(tb_directory, exist_ok=True)

    tensorboard = TensorBoard(log_dir=tb_directory)

    ###############################################################################


    ###############################################################################
    # Set up HPO.
    ###############################################################################

    config_file = "config.json"

    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            json_obj = json.load(f)
            prev_periods = int(json_obj["prev_periods"])
            #dropout_rate = float(json_obj["dropout_rate"])
    else:
        prev_periods = 1
        #dropout_rate = 0.2

    dropout_rate = 0.2
    # data_url = "https://ibm.box.com/shared/static/ojkntksc9rdbrj52yzkqfhbc1c9kv833.csv"

    data = pd.read_csv(input_data_path, index_col='DATE')

    # Create a scaled version of the data with oil prices normalized between 0 and 1
    values = data['WCOILWTICO'].values.reshape(-1,1)
    values = values.astype('float32')
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaled = scaler.fit_transform(values)
    # turn off scaler to simplify running model on future data
    scaled = values

    # Split the data between training and testing
    # The first 70% of the data is used for training while the remaining 30% is used for validation
    train_size = int(len(scaled) * 0.7)
    test_size = len(scaled) - train_size
    train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]

    # Generate testing and validation data
    trainX, trainY = gen_datasets(train, prev_periods)
    testX, testY = gen_datasets(test, prev_periods)

    # Reshape into a numpy arraya of shape (m, 1, prev_periods) where m is the number of training or testing values
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Build model
    lstm_units = 1000
    epochs = 20
    batch_size = 32
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(trainX.shape[1], trainX.shape[2])))
    if dropout_rate > 0.0:
       model.add(Dropout(dropout_rate))
    model.add(Dense(1))


    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    hpo = HPOMetrics()

    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY),  callbacks=[tensorboard, hpo], shuffle=False)

    hpo.close()
    #history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY),  callbacks=[tensorboard], shuffle=False)

    print("Training history:" + str(history.history))

    # Check out MSE, RMSE, MAE for  testing data
    testing_error = model.evaluate(testX, testY, verbose=0)
    print('Testing error: %.5f MSE (%.5f RMSE) %.5f MAE' % (testing_error[0], sqrt(testing_error[0]), testing_error[1]))


    # save the model
    model.save(output_model_path)
