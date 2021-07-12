import sys
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from utilities import multiply_durations, concat_all, get_max_packet_size, extract_packet_sizes, extract_packet_size_ranges, token_frequency_features_and_labels, packet_size_features, generate_traffic_rate_features, splitAllFeatures, extract_durations, normalize_packet_sizes
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import random
import csv

def extractSequences(fn):
    seqs = []
    with open(fn, newline='\n') as csvf:
        csv_reader = csv.reader(csvf, delimiter=' ')
        for row in csv_reader:
            if len(row) == 20:
                seqs.append(row)
    return seqs

def durationsToTimestamp(all_durations, max_duration=1.0):
    all_timestamps = []
    for durations in all_durations:
        timestamps = []
        float_durations = [float(x) for x in durations]
        for i in range(len(float_durations)):
            timestamps.append(sum(float_durations[0:i+1]) * max_duration)
        all_timestamps.append(timestamps)
    return all_timestamps


sys.setrecursionlimit(1000000)


warnings.filterwarnings('ignore')

# fix random seed for reproducibility
np.random.seed(7)

featuresFilePath = sys.argv[1]

with open(featuresFilePath, mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

with open("syntheticPreprocessed.pkl", mode='rb') as featuresFile:
    dg_synthetic_features = pickle.load(featuresFile)

device_to_packet_sizes = dict()
traffic_rate_feats = dict()
devices = []
raw_features = splitAllFeatures(raw_features)
raw_synthetic_features = splitAllFeatures(dg_synthetic_features)
device_to_timestamps = dict()
all_max_duration = 0
all_keys = []

for key, value in raw_features.items():
    all_keys.append(key)
    device_to_packet_sizes[key] = extract_packet_sizes(value)
    device_to_packet_sizes["dg-fake"+ key] = extract_packet_sizes(dg_synthetic_features[key])
    # traffic_rate_feats[key] = generate_traffic_rate_features(value)
    # traffic_rate_feats["dg-fake"+ key] = generate_traffic_rate_features(dg_synthetic_features[key])
    all_max_duration = max(all_max_duration, np.array(extract_durations(value)).max())

print(all_keys)

for key, value in raw_features.items():
    device_to_timestamps[key] = durationsToTimestamp(extract_durations(value))
    device_to_timestamps["dg-fake"+ key] = durationsToTimestamp(extract_durations(dg_synthetic_features[key]))

for key, value in raw_features.items():
    traffic_rate_feats[key] = generate_traffic_rate_features(concat_all(device_to_packet_sizes[key], device_to_timestamps[key]))
    traffic_rate_feats["dg-fake"+ key] = generate_traffic_rate_features(concat_all(device_to_packet_sizes["dg-fake"+ key], device_to_timestamps["dg-fake"+ key]))


rangeTokens, rangesToTokens, tokensToRanges, max_packet_size = extract_packet_size_ranges(device_to_packet_sizes)

def seq_discriminator(n_classes=2, max_length=20):
    in_seqs = Input(shape=(max_length,))
    inp_s = Embedding((max_packet_size * 2) + 1, 300, input_length=max_length)(in_seqs)
    inp_s = LSTM(max_length)(inp_s)
    out1 = Dense(n_classes, activation='softmax')(inp_s)
    model = Model([in_seqs], out1)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
    return model

all_accuracies = dict()
all_accuracies_knn = dict()

def randomize(selected, replace, size):
    copySelected = selected.copy()
    sampled = random.sample(copySelected, replace)
    for samp in sampled:
        copySelected.remove(samp)
    for i in range(replace):
        select = random.randint(0, size - 1)
        while select in copySelected:
            select = random.randint(0, size - 1)
        copySelected.add(select)
    return copySelected

for key in all_keys:
    max_packet_size = get_max_packet_size(device_to_packet_sizes[key])
    max_duration_file = key[:-7] + "max_duration.pkl"
    iotg_generated_durations_file = key[:-7] + "generated_durations.pkl"
    iotg_generated_packet_sizes_file = key[:-7] + "final_generated_packet_sizes.txt"
    print(max_duration_file)
    if not os.path.isfile(max_duration_file):
        continue
    with open(max_duration_file, mode='rb') as file:
        max_duration = float(pickle.load(file))
    with open(iotg_generated_durations_file, mode='rb') as file:
        iotg_generated_durations = pickle.load(file)
    iotg_generated_packets = extractSequences(iotg_generated_packet_sizes_file)
    # synthetic_iotg_packets = normalize_packet_sizes(iotg_generated_packets, max_packet_size=(max_packet_size * -1))[0]
    duration_counter = 0
    iotg_partitioned_durations = []
    for packet_seq in iotg_generated_packets:
        partitioned_durations = []
        for p in packet_seq:
            partitioned_durations.append(iotg_generated_durations[duration_counter])
            duration_counter += 1
        iotg_partitioned_durations.append(partitioned_durations)

    snythetic_iotg_tuples = concat_all(iotg_generated_packets, durationsToTimestamp(iotg_partitioned_durations, max_duration=max_duration))
    iot_tr_features = generate_traffic_rate_features(snythetic_iotg_tuples)
    currentRangeTokens = dict()
    # currentRangeTokens[key] = rangeTokens[key]
    # currentRangeTokens["dg-fake" + key] = rangeTokens["dg-fake" + key]
    currentDeviceToPacketSizes = dict()
    currentDeviceToPacketSizes[key] = device_to_packet_sizes[key]
    currentDeviceToPacketSizes["dg-fake" + key] = device_to_packet_sizes["dg-fake" + key]
    timestamps = device_to_timestamps[key]
    packet_sizes = normalize_packet_sizes(device_to_packet_sizes[key], max_packet_size=max_packet_size)[0]
    real_tr_feats = generate_traffic_rate_features(concat_all(packet_sizes, timestamps))

    current_metric = 100
    maxMetric = 0.05
    counter = 0
    replace_v = 1
    t_size = len(iot_tr_features)
    # real_tr_feats = traffic_rate_feats[key]

    selected_features_idx = set()
    for i in range(len(raw_features)):
        selected = random.randint(0, len(iot_tr_features) - 1)
        while selected in selected_features_idx:
            selected = random.randint(0, len(iot_tr_features) - 1)
        selected_features_idx.add(selected)

    print(key)
    print(iot_tr_features[0:10])
    print(real_tr_feats[0:10])

    while current_metric > maxMetric:
        counter += 1
        new_selected = randomize(selected_features_idx, replace_v, t_size)
        selected_gen_features = []
        for s in new_selected:
            selected_gen_features.append(iot_tr_features[s])
        features = real_tr_feats + selected_gen_features
        labels = [0] * len(real_tr_feats) + [1] * len(selected_gen_features)
        features = np.array(features)
        labels = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X_train, y_train)

        predicted = neigh.predict(X_test)
        score = accuracy_score(y_test, predicted)
        result_metric = abs(0.5 - score)
        if result_metric < current_metric:
            current_metric = result_metric
            print(current_metric)
            selected_features_idx = new_selected

    iot_gr_selected = []
    iot_gr_not_selected = []

    for i in range(len(iot_tr_features)):
        if i in s:
            iot_gr_selected.append(iot_tr_features[i])
        else:
            iot_gr_not_selected.append(iot_tr_features[i])

    sequence_feats = packet_size_features(currentDeviceToPacketSizes, max_packet_size)
    total_tokens = len(tokensToRanges)
    features, class_labels, total_classes, device_ids, max_sequence_length, devices = token_frequency_features_and_labels(
        currentRangeTokens, rangesToTokens)
    sequence_feats = sequence.pad_sequences(np.array(sequence_feats), maxlen=max_sequence_length)
    if len(currentRangeTokens["dg-fake" + key]) < 2:
        continue
    X_train, X_test, tr_train, tr_test, seq_train, seq_test, class_train, class_test = train_test_split(np.array(features), np.array(traffic_rate_feats[key] + traffic_rate_feats["fake" + key]), sequence_feats, np.array(class_labels), stratify=np.array(class_labels))
    discriminator = seq_discriminator()
    discriminator.summary()
    plot_model(discriminator, to_file='signature_discriminator_plot.png', show_shapes=True, show_layer_names=True)
    discriminator.fit([seq_train], class_train, epochs=500, batch_size=256, verbose=1)
    predictions = discriminator.predict([seq_test])
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

    accuracy = correct / (correct + wrong)
    print(key)
    print("Accuracy of model is ")

    print(accuracy)
    all_accuracies[key] = accuracy

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(tr_train, class_train)

    id_to_device = {v: k for k, v in device_ids.items()}
    # means
    tr_x = []
    # standard deviations
    tr_y = []
    groups_x = dict()
    groups_y = dict()
    for i in range(len(tr_train)):
        row = tr_train[i]
        groups_x[class_train[i]] = groups_x.get(class_train[i], []) + [row[0]]
        groups_y[class_train[i]] = groups_y.get(class_train[i], []) + [row[1]]
    groups_x_selected = []
    groups_y_selected = []
    for i in range(len(iot_gr_selected)):
        row = iot_gr_selected[i]
        groups_x_selected += [row[0]]
        groups_y_selected += [row[1]]
    groups_x_nonselected = []
    groups_y_nonselected = []
    for i in range(len(iot_gr_not_selected)):
        row = iot_gr_not_selected[i]
        groups_x_selected += [row[0]]
        groups_y_selected += [row[1]]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for id, device in id_to_device.items():
        if "fake" in device:
            color = "red"
            group = "DG synthetic"
            marker = "V"
        else:
            color = "blue"
            group = "real"
            marker = "."
        ax.scatter(np.array(groups_x[id]), np.array(groups_y[id]), alpha=0.8, c=color, marker = marker, edgecolors='none', s=30, label=group)

    ax.scatter(np.array(groups_x_selected), np.array(groups_y_selected), alpha=0.8, color="green", marker = ">", edgecolors='none', s=30, label="IoT Gen Selected")
    ax.scatter(np.array(groups_x_nonselected), np.array(groups_y_nonselected), alpha=0.8, color="yellow", marker = "<", edgecolors='none', s=30, label="IoT Gen Excluded")

    plt.title(key + ' scatter plot')
    plt.legend(loc=2)
    plt.xlabel('means')
    plt.ylabel('Mean of standard deviations')
    plt.savefig(key + 'scatterplot.png')

    predicted = neigh.predict(tr_test)
    score = accuracy_score(class_test, predicted)

    all_accuracies_knn[key] = score

with open("accuracies.pkl", mode='wb') as featureOutputFile:
    pickle.dump(all_accuracies, featureOutputFile)

with open("knnaccuracies.pkl", mode='wb') as featureOutputFile:
    pickle.dump(all_accuracies_knn, featureOutputFile)

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

# discriminator = signature_frequency_discriminator()
# discriminator.summary()
# plot_model(discriminator, to_file='signature_discriminator_plot.png', show_shapes=True, show_layer_names=True)
# discriminator.fit([X_train, tr_train, seq_train], class_train, epochs=500, batch_size=256, verbose=1)
# predictions = discriminator.predict([X_test, tr_test, seq_test])
# correct = 0
# wrong = 0
# predicted = []
# for i in range(len(predictions)):
#     predicted_class = np.argmax(predictions[i])
#     predicted.append(predicted_class)
#     target_class = class_test[i]
#     if predicted_class == target_class:
#         correct += 1
#     else:
#         wrong += 1
# accuracy = correct/(correct + wrong)
# print("Accuracy of model is ")
# print(accuracy)
# results = confusion_matrix(class_test, predicted)
# print(results)
# df_cm = pd.DataFrame(results, index = devices,
#                   columns = devices)
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()
#
# generated_features = list()
#
# selected_features_idx = set()
# for i in range(len(real_features)):
#     selected = random.randint(0, len(generated_features) - 1)
#     while selected in selected_features_idx:
#         selected = random.randint(0, len(generated_features) - 1)
#     selected_features_idx.add(selected)
#
# current_metric = 100
# maxMetric = 0.15
# counter = 0
# replace_v = 30
# t_size = len(generated_features)
#
# def randomize(selected, replace, size):
#     copySelected = selected.copy()
#     sampled = random.sample(copySelected, replace)
#     for samp in sampled:
#         copySelected.remove(samp)
#     for i in range(replace):
#         select = random.randint(0, size - 1)
#         while select in copySelected:
#             select = random.randint(0, size - 1)
#         copySelected.add(select)
#     return copySelected
#
# print("real")
# print(np.array(real_features[:20]))
# print("fake")
# print(np.array(generated_features[:20]))
#
# while current_metric > maxMetric:
#     counter += 1
#     new_selected = randomize(selected_features_idx, replace_v, t_size)
#     selected_gen_features = []
#     for s in new_selected:
#         selected_gen_features.append(generated_features[s])
#     features = real_features + selected_gen_features
#     labels = [0] * len(real_features) + [1] * len(selected_gen_features)
#     features = np.array(features)
#     labels = np.array(labels)
#
#     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)
#     neigh = KNeighborsClassifier(n_neighbors=5)
#     neigh.fit(X_train, y_train)
#
#     predicted = neigh.predict(X_test)
#     score = accuracy_score(y_test, predicted)
#     result_metric = abs(0.5 - score)
#     if result_metric < current_metric:
#         current_metric = result_metric
#         print(current_metric)
#         selected_features_idx = new_selected