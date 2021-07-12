import sys
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.preprocessing import sequence
from utilities import extract_packet_sizes, extract_packet_size_ranges, token_frequency_features_and_labels, packet_size_features, generate_traffic_rate_features, splitAllFeatures, extract_durations
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from gan.output import Output, OutputType, Normalization
import pathlib
import csv
sys.setrecursionlimit(1000000)


warnings.filterwarnings('ignore')

# fix random seed for reproducibility
np.random.seed(7)

featuresFilePath = sys.argv[1]

with open(featuresFilePath, mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

with open("synthetic.pkl", mode='rb') as featuresFile:
    synthetic_features = pickle.load(featuresFile)

device_to_packet_sizes = dict()
traffic_rate_feats = []
devices = []
raw_features = splitAllFeatures(raw_features)
raw_features["fake"] = synthetic_features["fake"]
device_to_durations = dict()
max_duration = 0

for key, value in raw_features.items():
    device_to_packet_sizes[key] = extract_packet_sizes(value)
    traffic_rate_feats += generate_traffic_rate_features(value)
    max_duration = max(max_duration, np.array(extract_durations(value)).max())

# packet_size_set = set()
#
# for key, value in device_to_packet_sizes.items():
#     for sequence in value:
#         for token in sequence:
#             packet_size_set.app

for key, value in raw_features.items():
    device_to_durations[key] = extract_durations(value, max_duration=max_duration)

rangeTokens, rangesToTokens, tokensToRanges, max_packet_size = extract_packet_size_ranges(device_to_packet_sizes)
sequence_feats = packet_size_features(device_to_packet_sizes, max_packet_size)
total_tokens = len(tokensToRanges)
features, class_labels, total_classes, device_ids, max_sequence_length, devices = token_frequency_features_and_labels(rangeTokens, rangesToTokens)
sequence_feats = sequence.pad_sequences(np.array(sequence_feats), maxlen=max_sequence_length)
X_train, X_test, tr_train, tr_test, seq_train, seq_test, class_train, class_test = train_test_split(np.array(features), np.array(traffic_rate_feats), sequence_feats, np.array(class_labels), stratify=np.array(class_labels))

for key, item in rangeTokens.items():
    dir = key + "/" + "real" + "/"
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    filename = dir + 'real_data.txt'
    with open(filename, mode='a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ')
        for seq in device_to_packet_sizes[key]:
            csv_writer.writerow(seq)

for key, item in device_to_durations.items():
    dir = key + "/" + "real" + "/"
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    filename = dir + 'real_durations.pkl'
    with open(filename, mode='wb') as durationFile:
        pickle.dump(item, durationFile)

with open("deviceToIds.pkl", mode="wb") as tokenFile:
    pickle.dump(device_ids, tokenFile)

with open("tokensToRanges.pkl", mode="wb") as tokenFile:
    pickle.dump(tokensToRanges, tokenFile)

with open("max_duration.pkl", mode="wb") as tokenFile:
    pickle.dump(max_duration, tokenFile)

with open("max_packet_Size.pkl", mode="wb") as tokenFile:
    pickle.dump(max_packet_size, tokenFile)

# V = (max_packet_size * 2) + 1
#
# data_feature = []
# data_attribute = []
# data_gen_flag = []
#
# total_devices = len(devices)
#
# device_number = 0
# for device, packet_sizes in device_to_packet_sizes.items():
#     normalized_durations = device_to_durations[device]
#     for i in range(len(packet_sizes)):
#         packet_sequence = packet_sizes[i]
#         duration_sequence = normalized_durations[i]
#         data_gen = []
#         data_feat = []
#         data_attr = [0] * total_devices
#         data_attr[device_number] = 1.0
#         for j in range(len(packet_sequence)):
#             direction = []
#             if packet_sequence[j] < 0:
#                 direction = [0.0, 1.0]
#             else:
#                 direction = [1.0, 0.0]
#             packet_size = float(abs(packet_sequence[j]))/float(max_packet_size)
#             normalized_duration = duration_sequence[j]
#             data_gen.append(1.0)
#             d = [packet_size]
#             d = d + direction
#             d.append(normalized_duration)
#             data_feat.append(np.array(d, dtype="float32"))
#         data_gen_flag.append(np.array(data_gen, dtype="float32"))
#         data_feature.append(np.array(data_feat))
#         data_attribute.append(np.array(data_attr, dtype="float32"))
#     device_number += 1
#
# data_feature_output = [
#     Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False),
#     Output(type_=OutputType.DISCRETE, dim=2, normalization=None, is_gen_flag=False),
#     Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False)
# ]
#
# data_attribute_output = [
#     Output(type_=OutputType.DISCRETE, dim=device_number, normalization=None, is_gen_flag=False)
# ]
#
# data_feature = np.array(data_feature)
# print(data_feature.shape)
# data_attribute = np.array(data_attribute)
# print(data_attribute.shape)
# data_gen_flag = np.array(data_gen_flag)
# print(data_gen_flag.shape)
#
# np.savez("data/iot/data_train.npz", data_feature=data_feature, data_attribute=data_attribute, data_gen_flag=data_gen_flag)
# with open('data/iot/data_feature_output.pkl', mode='wb') as fp:
#     pickle.dump(data_feature_output, fp, protocol=2)
# with open('data/iot/data_attribute_output.pkl', mode='wb') as fp:
#     pickle.dump(data_attribute_output, fp, protocol=2)

def signature_frequency_discriminator(tokenCount=total_tokens, n_classes=total_classes, max_length=max_sequence_length):
    in_frequencies = Input(shape=(tokenCount,))
    in_guassians = Input(shape=(4,))
    in_seqs = Input(shape=(max_length,))
    inp_s = Embedding((max_packet_size * 2) + 1, 300, input_length=max_length)(in_seqs)
    inp_s = LSTM(300)(inp_s)
    inp_s = Dense(100, activation='relu')(inp_s)
    inp_s = Dense(100, activation='relu')(inp_s)
    inp_g = Dense(100, activation='relu')(in_guassians)
    inp_g = Dense(100, activation='relu')(inp_g)
    inp_g = Dense(100, activation='relu')(inp_g)
    inp_g = Dense(100, activation='relu')(inp_g)
    inp_g = Dense(100, activation='relu')(inp_g)
    inp = Dense(1000, activation='relu')(in_frequencies)
    inp = Dense(500, activation='relu')(inp)
    inp = Dense(250, activation='relu')(inp)
    inp = Dense(250, activation='relu')(inp)
    inp = Dense(250, activation='relu')(inp)
    inp = Dense(250, activation='relu')(inp)
    inp = Dense(250, activation='relu')(inp)
    merged = Concatenate()([inp, inp_g, inp_s])
    out = Dense(500, activation='relu')(merged)
    out = Dense(400, activation='relu')(out)
    out = Dense(300, activation='relu')(out)
    out = Dense(200, activation='relu')(out)
    out = Dense(200, activation='relu')(out)
    out = Dense(200, activation='relu')(out)
    out = Dense(200, activation='relu')(out)
    out = Dense(200, activation='relu')(out)
    out = Dense(200, activation='relu')(out)
    out = Dense(150, activation='relu')(out)
    out = Dense(100, activation='relu')(out)
    out = Dense(50, activation='relu')(out)
    out1 = Dense(n_classes, activation='softmax')(out)
    model = Model([in_frequencies, in_guassians, in_seqs], out1)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
    return model

discriminator = signature_frequency_discriminator()
discriminator.summary()
plot_model(discriminator, to_file='signature_discriminator_plot.png', show_shapes=True, show_layer_names=True)
discriminator.fit([X_train, tr_train, seq_train], class_train, epochs=500, batch_size=256, verbose=1)
predictions = discriminator.predict([X_test, tr_test, seq_test])
correct = 0
wrong = 0
predicted = []
for i in range(len(predictions)):
    predicted_class = np.argmax(predictions[i])
    predicted.append(predicted_class)
    target_class = class_test[i]
    if predicted_class == target_class:
        correct += 1
    else:
        wrong += 1
accuracy = correct/(correct + wrong)
print("Accuracy of model is ")
print(accuracy)
results = confusion_matrix(class_test, predicted)
print(results)
df_cm = pd.DataFrame(results, index = devices,
                  columns = devices)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()
