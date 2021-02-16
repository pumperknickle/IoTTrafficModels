import sys
from utilities import directory_packet_feature_extraction
import pickle

directory = sys.argv[1]
output = sys.argv[2]

features = directory_packet_feature_extraction(directory)
with open(output, mode='wb') as featureOutputFile:
    pickle.dump(features, featureOutputFile)
