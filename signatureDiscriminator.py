import sys
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Concatenate
from utilities import extract_packet_sizes, extract_packet_size_ranges, token_frequency_features_and_labels, normalize_durations, generate_traffic_rate_features, splitAllFeatures, extract_signature_frequencies, signature_frequencies_features, extract_durations
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import csv
import pathlib

warnings.filterwarnings('ignore')

# fix random seed for reproducibility
np.random.seed(7)

featuresFilePath = sys.argv[1]

with open(featuresFilePath, mode='rb') as featuresFile:
    features = pickle.load(featuresFile)

device_to_packet_sizes = dict()
traffic_rate_feats = []
devices = []
print(features)
features = splitAllFeatures(features)
device_to_durations = dict()
all_durations = []

for key, value in features.items():
    device_to_packet_sizes[key] = extract_packet_sizes(value)
    traffic_rate_feats += generate_traffic_rate_features(value)
    device_to_durations[key] = extract_durations(value)
    all_durations.append(device_to_durations[key])

freqs, numberOfSigs, max_packet_size = extract_signature_frequencies(device_to_packet_sizes)
freq_features, class_labels, total_classes, device_ids, devices = signature_frequencies_features(freqs)

X_train, X_test, tr_train, tr_test, class_train, class_test = train_test_split(np.array(freq_features), np.array(traffic_rate_feats), np.array(class_labels), stratify=np.array(class_labels))

rangeTokens, rangesToTokens, tokensToRanges, max_packet_size = extract_packet_size_ranges(device_to_packet_sizes)
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
    filename = dir + 'real_durations.txt'
    with open(filename, mode='wb') as durationFile:
        pickle.dump(item, durationFile)




def signature_frequency_discriminator(sigNumber=numberOfSigs, n_classes=total_classes):
    in_frequencies = Input(shape=(sigNumber,))
    in_guassians = Input(shape=(4,))
    # in_seqs = Input(shape=(max_length,))
    # inp_s = Embedding(tokenCount + 1, 32, input_length=max_length)(in_seqs)
    # inp_s = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(inp_s)
    # inp_s = MaxPooling1D(pool_size=2)(inp_s)
    # inp_s = Dropout(0.95)(inp_s)
    # inp_s = LSTM(max_length)(inp_s)
    # inp_s = Dense(100, activation='relu')(inp_s)
    # inp_s = Dense(100, activation='relu')(inp_s)
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
    merged = Concatenate()([inp, inp_g])
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
    model = Model([in_frequencies, in_guassians], out1)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
    return model

discriminator = signature_frequency_discriminator()
discriminator.summary()
plot_model(discriminator, to_file='signature_discriminator_plot.png', show_shapes=True, show_layer_names=True)
discriminator.fit([X_train, tr_train], class_train, epochs=1000, batch_size=256, verbose=1)
predictions = discriminator.predict([X_test, tr_test])
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


