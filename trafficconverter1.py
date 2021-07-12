import pickle
import numpy as np

firstSet = np.load("first.npz")
secondSet = np.load("second.npz")

with open("preprocessed.pkl", mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

with open("max_packet_Size.pkl", mode='rb') as packetSizeFile:
    max_packet_size = pickle.load(packetSizeFile)

with open("deviceToIds.pkl", mode='rb') as packetSizeFile:
    deviceToIds = pickle.load(packetSizeFile)

idToDevices = {v: k for k, v in deviceToIds.items()}

all_sequences = []
sequencesForDevices = dict()
for i in range(len(firstSet['features'])):
    featureSets = firstSet['features'][i]
    labelSet = firstSet['attributes'][i]
    label = np.argmax(labelSet)
    sequence = []
    for featureSet in featureSets:
        packet_size = round(featureSet[0] * float(max_packet_size))
        direction = np.argmax(featureSet[1:3])
        if direction is 1:
            packet_size = -1 * packet_size
        duration = featureSet[3]
        sequence.append([packet_size, duration])
    all_sequences.append(sequence)
    sequencesForDevices[idToDevices[label]] = sequencesForDevices.get(idToDevices[label], []) + [sequence]

for i in range(len(secondSet['features'])):
    featureSets = secondSet['features'][i]
    labelSet = secondSet['attributes'][i]
    label = np.argmax(labelSet)
    sequence = []
    for featureSet in featureSets:
        packet_size = round(featureSet[0] * float(max_packet_size))
        direction = np.argmax(featureSet[1:3])
        if direction is 1:
            packet_size = -1 * packet_size
        duration = featureSet[3]
        sequence.append([packet_size, duration])
    all_sequences.append(sequence)
    sequencesForDevices[idToDevices[label]] = sequencesForDevices.get(idToDevices[label], []) + [sequence]

synthetic_features = dict()

synthetic_features["fake"] = all_sequences

with open("synthetic.pkl", mode='wb') as featureOutputFile:
    pickle.dump(raw_features, featureOutputFile)

with open("syntheticPreprocessed.pkl", mode='wb') as featureOutputFile:
    pickle.dump(sequencesForDevices, featureOutputFile)