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
from statistics import multimode

def durationsToTimestamp(all_durations, max_duration=1.0):
    all_timestamps = []
    for durations in all_durations:
        timestamps = []
        float_durations = [float(x) for x in durations]
        for i in range(len(float_durations)):
            timestamps.append(sum(float_durations[0:i+1]) * max_duration)
        all_timestamps.append(timestamps)
    return all_timestamps

def flatten(features):
    flattened = []
    for feature in features:
        for f in feature:
            flattened.append(f)
    return flattened

sys.setrecursionlimit(1000000)

warnings.filterwarnings('ignore')

# fix random seed for reproducibility
np.random.seed(7)

featuresFilePath = sys.argv[1]

with open(featuresFilePath, mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

raw_features = splitAllFeatures(raw_features)
all_keys = []
features = []
labels = []
id_to_device = dict()

counter = 0
for key, value in raw_features.items():
    id_to_device[key] = counter
    all_keys.append(key)
    traffic_rate_feats = generate_traffic_rate_features(concat_all(extract_packet_sizes(value), durationsToTimestamp(extract_durations(value))))
    features += traffic_rate_feats
    labels += [counter] * len(traffic_rate_feats)
    counter += 1

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42, stratify=labels)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(np.array(X_train), np.array(y_train))

correct = 0.0

for i in range(len(X_test)):
    test_feat = X_test[i]
    test_label = y_test[i]
    predicted = neigh.predict([test_feat])
    if test_label == predicted:
        correct += 1

print("accuracy")
print(correct/float(len(X_test)))


