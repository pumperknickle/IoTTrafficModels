import pickle
import numpy as np

firstSet = np.load("first.npz")
secondSet = np.load("second.npz")
thirdSet = np.load("third.npz")
fourthSet = np.load("fourth.npz")
fifthSet = np.load("fifth.npz")
sixthSet = np.load("sixth.npz")


with open("preprocessed.pkl", mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

with open("max_packet_Size.pkl", mode='rb') as packetSizeFile:
    max_packet_size = pickle.load(packetSizeFile)

with open("deviceToIds.pkl", mode='rb') as packetSizeFile:
    deviceToIds = pickle.load(packetSizeFile)

idToDevices = {v: k for k, v in deviceToIds.items()}

sequencesForDevices = dict()

all_sequences = []
for i in range(len(firstSet['features'])):
    featureSets = firstSet['features'][i]
    labelSet = firstSet['attributes'][i]
    label = np.argmax(labelSet)
    sequence = []
    for featureSet in featureSets:
        packet_size = int(round(featureSet[0] * float(max_packet_size)))
        direction = np.argmax(featureSet[1:3])
        if direction == 1:
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
        packet_size = int(round(featureSet[0] * float(max_packet_size)))
        direction = np.argmax(featureSet[1:3])
        if direction == 1:
            packet_size = -1 * packet_size
        duration = featureSet[3]
        sequence.append([packet_size, duration])
    all_sequences.append(sequence)
    sequencesForDevices[idToDevices[label]] = sequencesForDevices.get(idToDevices[label], []) + [sequence]

for i in range(len(thirdSet['features'])):
    featureSets = thirdSet['features'][i]
    labelSet = thirdSet['attributes'][i]
    label = np.argmax(labelSet)
    sequence = []
    for featureSet in featureSets:
        packet_size = int(round(featureSet[0] * float(max_packet_size)))
        direction = np.argmax(featureSet[1:3])
        if direction == 1:
            packet_size = -1 * packet_size
        duration = featureSet[3]
        sequence.append([packet_size, duration])
    all_sequences.append(sequence)
    sequencesForDevices[idToDevices[label]] = sequencesForDevices.get(idToDevices[label], []) + [sequence]

for i in range(len(fourthSet['features'])):
    featureSets = fourthSet['features'][i]
    labelSet = fourthSet['attributes'][i]
    label = np.argmax(labelSet)
    sequence = []
    for featureSet in featureSets:
        packet_size = int(round(featureSet[0] * float(max_packet_size)))
        direction = np.argmax(featureSet[1:3])
        if direction == 1:
            packet_size = -1 * packet_size
        duration = featureSet[3]
        sequence.append([packet_size, duration])
    all_sequences.append(sequence)
    sequencesForDevices[idToDevices[label]] = sequencesForDevices.get(idToDevices[label], []) + [sequence]

for i in range(len(fifthSet['features'])):
    featureSets = fifthSet['features'][i]
    labelSet = fifthSet['attributes'][i]
    label = np.argmax(labelSet)
    sequence = []
    for featureSet in featureSets:
        packet_size = int(round(featureSet[0] * float(max_packet_size)))
        direction = np.argmax(featureSet[1:3])
        if direction == 1:
            packet_size = -1 * packet_size
        duration = featureSet[3]
        sequence.append([packet_size, duration])
    all_sequences.append(sequence)
    sequencesForDevices[idToDevices[label]] = sequencesForDevices.get(idToDevices[label], []) + [sequence]

for i in range(len(sixthSet['features'])):
    featureSets = sixthSet['features'][i]
    labelSet = sixthSet['attributes'][i]
    label = np.argmax(labelSet)
    sequence = []
    for featureSet in featureSets:
        packet_size = int(round(featureSet[0] * float(max_packet_size)))
        direction = np.argmax(featureSet[1:3])
        if direction == 1:
            packet_size = -1 * packet_size
        duration = featureSet[3]
        sequence.append([packet_size, duration])
    all_sequences.append(sequence)
    sequencesForDevices[idToDevices[label]] = sequencesForDevices.get(idToDevices[label], []) + [sequence]

synthetic_features = dict()

synthetic_features["fake"] = all_sequences

with open("synthetic.pkl", mode='wb') as featureOutputFile:
    pickle.dump(synthetic_features, featureOutputFile)

with open("syntheticPreprocessed.pkl", mode='wb') as featureOutputFile:
    pickle.dump(sequencesForDevices, featureOutputFile)
