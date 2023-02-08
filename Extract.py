from lightgbm import train
import numpy as np
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score
import keras
from keras.models import Model, load_model
from keras.layers import Dense, Bidirectional, LSTM, Permute, Input, Flatten, multiply, Embedding, RepeatVector, \
    TimeDistributed, Dropout, Reshape
from keras.utils import np_utils
from matplotlib import pyplot as plt
import seaborn as sns
import time
import pandas as pd
from sklearn import preprocessing
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=193)
import logging

logging.disable(30)

# TIME_STEP为因序列长度，即word2vec数据的feature_len - 2
# TIME_STEP = 99

# 其他参数可以不修改
NUMCLASSES = 2
BATCH_SIZE = 64
EPOCHS = 50
EMBEDDING_DIM = 32
ENCODER_UNITS = 32
DECODER_UNITS = 16


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()

    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def confustion_matrix(labels, predicions):
    LABELS = ['POS', 'NEG']
    matrix = metrics.confustion_matrix(labels, predicions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, cmap='coolwarm', linecolor='white', linewidths=1, xticklabels=LABELS, yticklabels=LABELS,
                annot=True, fmt='d')
    plt.title('Conion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def get_TIME_STEP(dataset_name):
    path = './Data/word2vec/{}-P.txt'.format(dataset_name)
    with open(path, "r") as fr:
        line = fr.readline()
        return len(line.replace('\n', '')) - 2


def get_label(dataset_name):
    pos_path = "Data/PSEKNC/{}-P.npy".format(dataset_name)
    pos_data = np.load(pos_path)
    neg_path = "Data/PSEKNC/{}-N.npy".format(dataset_name)
    neg_data = np.load(neg_path)

    label = [1] * pos_data.shape[0] + [0] * neg_data.shape[0]
    return np.array(label)


def attention(decoder_out, TIME_STEP):
    weight = Permute((2, 1))(decoder_out)
    weight = Dense(TIME_STEP, activation='softmax')(weight)
    weight = Permute((2, 1))(weight)
    attention_output = multiply([decoder_out, weight])  # multiply(a,b)是个 a和 b的乘法。如果 a,b是两个数组，那么对应元素相乘。
    # 如果a和 b的shape不一样，就会采用广播。
    return attention_output


def seq2ngram(seq_path, k, s, model):
    with open(seq_path, "r") as fr:
        lines = fr.readlines()
    fr.close()
    list_full_text = []
    for line in lines:
        if line.startswith(">hsa") or len(line) <= 1:
            continue
        else:
            line = line[:-1].upper()
            seq_len = len(line)
            list_line = []
            for index in range(0, seq_len, s):
                if index + k >= seq_len + 1:
                    break
                list_line.append(line[index:index + k])
            word_index = []
            for word in list_line:
                if word in model.wv:
                    word_index.append(model.wv.vocab[word].index)
            list_full_text.append(word_index)
    return list_full_text


def train_net(data_mode, dataset_name):
    if data_mode == 'word2vec':
        return SeqNet(data_mode, dataset_name)
    else:
        return SeqNet_else(data_mode, dataset_name)


def SeqNet(data_mode, dataset_name):
    TIME_STEP = get_TIME_STEP(dataset_name)
    ########################################  WORD2VEC  #########################################################
    word2vec_model = word2vec.Word2Vec.load('E:/PythonProject/TRY/WCLAS模型/save_w2v_model/ZC3H7B.model')
    print(type(word2vec_model))

    pos_path = "Data/{}/{}-P.txt".format(data_mode, dataset_name)
    pos_list1 = seq2ngram(pos_path, 3, 1, word2vec_model)
    neg_path = "Data/{}/{}-N.txt".format(data_mode, dataset_name)
    neg_list1 = seq2ngram(neg_path, 3, 1, word2vec_model)

    seq_list1 = np.append(pos_list1, neg_list1, axis=0)

    feature = seq_list1
    print('feature.shape', feature.shape)
    label = [1] * len(pos_list1) + [0] * len(neg_list1)

    embedding_matrix = np.zeros((len(word2vec_model.wv.vocab), EMBEDDING_DIM))
    for i in range(len(word2vec_model.wv.vocab)):
        embedding_vector = word2vec_model.wv[word2vec_model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    inputs1 = Input(shape=(TIME_STEP,))
    embedding = Embedding(input_dim=embedding_matrix.shape[0],
                          output_dim=EMBEDDING_DIM,
                          weights=[embedding_matrix],
                          trainable=True)(inputs1)

    #  CNN 层
    cnn_out = Convolution1D(activation="relu", input_shape=(None, 32), filters=102, kernel_size=5)(embedding)
    pooling_out = MaxPooling1D(pool_size=5)(cnn_out)
    pooling_out = Dropout(0.2)(pooling_out)
    # Encoder layers
    encoder_out = Bidirectional(LSTM(ENCODER_UNITS, return_sequences=True))(pooling_out)
    encoder_out = Bidirectional(LSTM(ENCODER_UNITS, return_sequences=False))(encoder_out)
    encoder_out = RepeatVector(TIME_STEP)(encoder_out)
    # Decoder layers
    decoder_out = LSTM(DECODER_UNITS, return_sequences=True)(encoder_out)
    decoder_out = TimeDistributed(Dense(DECODER_UNITS // 2))(decoder_out)
    # Attention layer
    attention_out = attention(decoder_out, TIME_STEP)
    attention_out = Flatten()(attention_out)
    fc_count = Dense(DECODER_UNITS)(attention_out)
    fc_count = Dense(DECODER_UNITS // 2)(fc_count)
    output = Dense(units=2, activation='softmax')(fc_count)

    model = Model(inputs=[inputs1], outputs=output)

    print('model.summary', model.summary())

    print('\n****** Fit the model ******\n')
    time_callback = TimeHistory()
    callbacks_list = [
        time_callback,
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ]
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    label = np_utils.to_categorical(label, NUMCLASSES)
    model.fit(feature, label, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list, validation_split=0.2,
              verbose=1)
    save_dir = 'save_model(1)/{}'.format(data_mode)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    model.save(os.path.join(save_dir, "{}.h5".format(dataset_name)))


def SeqNet_else(data_mode, dataset_name):
    TIME_STEP = get_TIME_STEP(dataset_name)

    pos_path = "Data/{}/{}-P.npy".format(data_mode, dataset_name)
    pos_data = np.load(pos_path)
    neg_path = "Data/{}/{}-N.npy".format(data_mode, dataset_name)
    neg_data = np.load(neg_path)

    feature = np.vstack((pos_data, neg_data))
    feature = feature.reshape(feature.shape[0], feature.shape[1], 1)

    label = [1] * pos_data.shape[0] + [0] * neg_data.shape[0]

    inputs = Input(shape=(feature.shape[1], 1))

    #  CNN 层
    cnn_out = Convolution1D(activation="relu", input_shape=(None, feature.shape[1]), filters=102, kernel_size=5)(inputs)
    pooling_out = MaxPooling1D(pool_size=5)(cnn_out)
    pooling_out = Dropout(0.2)(pooling_out)
    # Encoder layers
    encoder_out = Bidirectional(LSTM(ENCODER_UNITS, return_sequences=True))(pooling_out)
    encoder_out = Bidirectional(LSTM(ENCODER_UNITS, return_sequences=False))(encoder_out)
    encoder_out = RepeatVector(TIME_STEP)(encoder_out)
    # Decoder layers
    decoder_out = LSTM(DECODER_UNITS, return_sequences=True)(encoder_out)
    decoder_out = TimeDistributed(Dense(DECODER_UNITS // 2))(decoder_out)
    # Attention layer
    attention_out = attention(decoder_out, TIME_STEP)
    attention_out = Flatten()(attention_out)
    fc_count = Dense(DECODER_UNITS)(attention_out)
    fc_count = Dense(DECODER_UNITS // 2)(fc_count)
    output = Dense(units=2, activation='softmax')(fc_count)

    model = Model(inputs=[inputs], outputs=output)

    print(model.summary())

    print('\n****** Fit the model ******\n')
    time_callback = TimeHistory()
    callbacks_list = [
        time_callback,
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ]
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    label = np_utils.to_categorical(label, NUMCLASSES)
    model.fit(feature, label, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list, validation_split=0.2,
              verbose=1)
    save_dir = 'save_model(1)/{}'.format(data_mode)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    model.save(os.path.join(save_dir, "{}.h5".format(dataset_name)))


def get_extraction_feature(data_mode, dataset_name):
    word2vec_model = word2vec.Word2Vec.load('E:/PythonProject/TRY/WCLAS模型/save_w2v_model/ZC3H7B.model')
    layer_size = 16

    if data_mode == 'word2vec':
        pos_path = "Data/{}/{}-P.txt".format(data_mode, dataset_name)
        pos_list1 = seq2ngram(pos_path, 3, 1, word2vec_model)
        neg_path = "Data/{}/{}-N.txt".format(data_mode, dataset_name)
        neg_list1 = seq2ngram(neg_path, 3, 1, word2vec_model)

        seq_list1 = np.append(pos_list1, neg_list1, axis=0)
        feature = seq_list1
    else:
        pos_path = "Data/{}/{}-P.npy".format(data_mode, dataset_name)
        pos_data = np.load(pos_path)
        neg_path = "Data/{}/{}-N.npy".format(data_mode, dataset_name)
        neg_data = np.load(neg_path)
        feature = np.vstack((pos_data, neg_data))
        feature = feature.reshape(feature.shape[0], feature.shape[1], 1)
        layer_size = 15

    model_path = 'save_model(1)/{}/{}.h5'.format(data_mode, dataset_name)
    if os.path.exists(model_path) == False:
        print('model not exit, training...')
        start = time.time()
        train_net(data_mode, dataset_name)
        end = time.time()
        print('model train finished, use time: {} s'.format(end - start))
    model = load_model(model_path)

    layer_model = Model(inputs=model.input, outputs=model.layers[layer_size].output)
    output = layer_model.predict(feature)
    return output


def combine(data_modes, dataset_name, is_save=True):
    features = []
    for data_mode in data_modes:
        feature = get_extraction_feature(data_mode, dataset_name)
        features.append(feature)

    res = np.hstack((features[0], features[1]))
    for i in range(2, len(features)):
        res = np.hstack((res, features[i]))
    save_path = 'Extract_result(1)/{}.npy'.format(dataset_name)
    labels = get_label(dataset_name)
    labels = labels.reshape(labels.shape[0], 1)
    print('****************************************', res.shape, labels.shape)
    res = np.hstack((res, labels))
    if is_save:
        np.save(save_path, res)


def main():
    # run the model
    DATA_MODES = ["word2vec", "PSEKNC", "PSTNP", "TNC"]
    dataset_names = ['ZC3H7B']
    for dataset_name in dataset_names:
        start = time.time()
        combine(DATA_MODES, dataset_name)
        end = time.time()
        print('{} is finish, use time: {}'.format(dataset_name, end - start))


def check():
    path = 'Extract_result(1)/ZC3H7B.npy'
    datas = np.load(path)
    print(datas.shape)
    print(datas)


if __name__ == "__main__":
    main()