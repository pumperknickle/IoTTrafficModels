import sys
import numpy as np
import pickle
import os
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from utilities import normalize_packet_sizes, extract_packet_sizes, extract_packet_size_ranges, token_frequency_features_and_labels, packet_size_features, generate_traffic_rate_features, splitAllFeatures, extract_durations, durationcluster, toDurationRanges
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import csv

sys.setrecursionlimit(1000000)

warnings.filterwarnings('ignore')

# fix random seed for reproducibility
np.random.seed(7)

featuresFilePath = sys.argv[1]

with open(featuresFilePath, mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

with open("max_packet_Size.pkl", mode='rb') as featuresFile:
    max_packet_size = pickle.load(featuresFile)

device_to_packet_sizes = dict()
devices = []
raw_features = splitAllFeatures(raw_features)
device_to_durations = dict()
max_duration = 0
all_keys = []
all_durations = []
new_raw = dict()

for key, value in raw_features.items():
    if len(value) > 20:
        new_raw[key] = value

raw_features = new_raw

for key, value in raw_features.items():
    all_keys.append(key)
    device_to_packet_sizes[key] = extract_packet_sizes(value)
    max_duration = max(max_duration, np.array(extract_durations(value)).max())

for key, value in raw_features.items():
    device_to_durations[key] = extract_durations(value, max_duration=max_duration)
    all_durations += list(np.array(device_to_durations[key]).flatten())

rangeTokens, rangesToTokens, tokensToRanges, max_packet_size = extract_packet_size_ranges(device_to_packet_sizes)
clusters = durationcluster(all_durations)
durationRangesToTokens, tokensToDurationRanges = toDurationRanges(clusters)

for key, value in raw_features.items():
    device_to_packet_sizes[key] = normalize_packet_sizes(extract_packet_sizes(value), max_packet_size)[0]

counter = 0
for key in all_keys:
    deviceString = "device_" + str(counter)
    if not os.path.exists(deviceString):
        os.makedirs(deviceString)
    with open(deviceString + "/" + "device_name.pkl", mode='wb') as nameFile:
        pickle.dump(key, nameFile)
    with open(deviceString + "/" + "train_y.pkl", mode='wb') as trainFile:
        pickle.dump(device_to_durations[key], trainFile)
    tokenForDevice = rangeTokens[key]
    with open(deviceString + "/" + "train_X.pkl", mode='wb') as tokenFile:
        pickle.dump(tokenForDevice, tokenFile)
    with open(deviceString + "/" + "durationRangesToToken.pkl", mode='wb') as rangeFile:
        pickle.dump(durationRangesToTokens, rangeFile)
    with open(deviceString + "/" + "rangesToToken.pkl", mode='wb') as rangeFile:
        pickle.dump(rangesToTokens, rangeFile)
    packet_sizes = device_to_packet_sizes[key]
    with open(deviceString + "/" + "real_packet_sizes.txt", mode='w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ')
        for chunk in packet_sizes:
            if len(chunk) == 20:
                new_list = [x for x in chunk]
                csv_writer.writerow(new_list)
    with open(deviceString + "/" + "real_data.txt", mode='w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ')
        for chunk in tokenForDevice:
            if len(chunk) == 20:
                new_list = [x for x in chunk]
                csv_writer.writerow(new_list)
    with open(deviceString + "/"+ "max_duration.pkl", mode='wb') as tokenFile:
        pickle.dump(max_duration, tokenFile)
    with open(deviceString + "/" + "max_packet_size.pkl", mode='wb') as tokenFile:
        pickle.dump(max_packet_size, tokenFile)
    counter += 1

