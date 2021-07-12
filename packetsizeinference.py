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
from utilities import dbclustermin, ngrams, extractFeatures, extractSignatures, concat_all, get_max_packet_size, extract_packet_sizes, extract_packet_size_ranges, token_frequency_features_and_labels, packet_size_features, generate_traffic_rate_features, splitAllFeatures, extract_durations, normalize_packet_sizes
import warnings
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import random
import csv

# distance metric used by dbscan
distance_threshold = 5.0
# total ngrams divided by cluster threshold is equal to the min_samples needed to form a cluster in dbscan
min_cluster = 4
min_sig_size = 2
max_sig_size = 5

sys.setrecursionlimit(1000000)

warnings.filterwarnings('ignore')

# fix random seed for reproducibility
np.random.seed(7)

featuresFilePath = sys.argv[1]

with open(featuresFilePath, mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

device_to_packet_sizes = dict()
raw_features = splitAllFeatures(raw_features)

all_keys = []

all_fake_features = dict()
all_real_features = dict()

packetinferenceresults = dict()
f1scores = dict()
f1normscores = dict()
mccscores = dict()

packet_size_min = 0
packet_size_max = 0

for key, value in raw_features.items():
    packet_size_sequences = extract_packet_sizes(value)
    for seq in packet_size_sequences:
        packet_size_min = min(min(seq), packet_size_min)
        packet_size_max = max(max(seq), packet_size_max)

for key, value in raw_features.items():
    all_keys.append(key)
    device_to_packet_sizes[key] = normalize_packet_sizes(extract_packet_sizes(value), max_packet_size=abs(packet_size_min))[0]

def baseline_model(tokenCount = packet_size_max + abs(packet_size_min) + 1, n_classes=2, max_length=20):
    in_seqs = Input(shape=(max_length,))
    inp_s = Embedding(tokenCount + 1, 32, input_length=max_length)(in_seqs)
    inp_s = LSTM(max_length)(inp_s)
    out1 = Dense(n_classes, activation='softmax')(inp_s)
    model = Model(in_seqs, out1)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
    return model

for key in all_keys:
    real = device_to_packet_sizes[key]
    fake = []
    for a in all_keys:
        if a == key:
            continue
        fake += device_to_packet_sizes[a]
    test_split_factor = 0.33
    total_fakes_needed = len(fake) * test_split_factor
    if len(real) * 2 < 100:
        extra_fake_features, fake = train_test_split(fake, test_size=100 / len(fake), random_state=42)
        if len(fake) * test_split_factor < total_fakes_needed:
            extra_fake_features = random.sample(extra_fake_features,
                                                int(total_fakes_needed - (len(fake) * test_split_factor)))
        else:
            extra_fake_features = []
    else:
        if len(real) * 2 < len(fake):
            extra_fake_features, fake = train_test_split(fake, test_size=len(real) * 2 / len(fake), random_state=42)
            if len(fake) * test_split_factor < total_fakes_needed:
                extra_fake_features = random.sample(extra_fake_features,
                                                    int(total_fakes_needed - (len(fake) * test_split_factor)))
            else:
                extra_fake_features = []
        else:
            extra_fake_features = []
    labels = np.array([0] * (len(real)) + ([1] * len(fake)))
    features = np.array(real + fake)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_split_factor, random_state=42)
    X_test = np.array(list(X_test) + extra_fake_features)
    y_test = np.array(list(y_test) + ([1] * len(extra_fake_features)))
    discriminator = baseline_model()
    discriminator.fit(X_train, y_train, epochs=500, batch_size=16, verbose=1)
    discriminator.summary()
    plot_model(discriminator, to_file='packetsizediscriminator.png', show_shapes=True, show_layer_names=True)
    predictions = discriminator.predict(X_test)
    correct = 0
    wrong = 0
    predicted = []
    positives = 0
    negatives = 0

    for i in range(len(predictions)):
        predicted_class = np.argmax(predictions[i])
        predicted.append(predicted_class)
        target_class = y_test[i]
        if target_class == 0:
            positives += 1
        else:
            negatives += 1
        if predicted_class == target_class:
            correct += 1
        else:
            wrong += 1

    accuracy = correct / (correct + wrong)
    print(key)
    print("Accuracy of model is ")
    print(accuracy)
    packetinferenceresults[key] = accuracy
    results = confusion_matrix(y_test, predicted)
    print(results)
    print("f1 score")
    f1 = f1_score(y_test, predicted, pos_label=0, average='binary')
    print(f1)
    f1scores[key] = f1
    q = positives / (positives + negatives)
    f1_coin = (2 * q) / (q + 1)
    f1_norm = (f1 - f1_coin) / (1 - f1_coin)
    print("f1 norm")
    print(f1_norm)
    f1normscores[key] = f1_norm

    mcc = matthews_corrcoef(y_test, predicted)
    print("mcc")
    print(mcc)
    mccscores[key] = mcc

with open("packetsizeinferenceresultsexperiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(packetinferenceresults, featureOutputFile)

with open("packetsizef1experiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(f1scores, featureOutputFile)

with open("packetsizef1normexperiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(f1normscores, featureOutputFile)

with open("packetsizemccscoresexp1.pkl", mode='wb') as featureOutputFile:
    pickle.dump(mccscores, featureOutputFile)
