import os
import numpy as np
from collections import Counter


# This is the class for extracting the features extracted from deep models
class deep_model_feature_reader:

    def __init__(self, file_path, data_mode, timechainParameter=True, one_hot=False, duplicate=False):

        self.features = []
        self.labels = []

        if duplicate == True:
            self.labels_amplify = {}

        if timechainParameter == True:
            self.dict_subpart_2_fullpart = {}
            self.dict_subpart_2_timechain = {}

        with open(file_path+data_mode+"_data.txt", "r+") as doc:
            for i,l in enumerate(doc):
                line = l.replace("\n", "").split("\t")
                feature = [float(x) for x in line[0].split(",")[:-1]]
                label = int(line[1])
                if timechainParameter == True:
                    fullpart = int(line[2])
                    timechain = int(line[3])
                    self.dict_subpart_2_fullpart[i] = fullpart
                    self.dict_subpart_2_timechain[i] = timechain
                self.features.append(feature)
                self.labels.append(label)

        if duplicate == True:
            cnt = Counter()
            for labels in self.labels:
                cnt[labels] += 1
            max_value = np.amax(list(cnt.values()))
            for labels in range(4):
                label_amplify = int(np.floor(max_value/cnt[labels]))
                self.labels_amplify[labels] = label_amplify
            new_features = []
            new_labels = []
            for i in range(len(self.features)):
                for _ in range(self.labels_amplify[self.labels[i]]):
                    new_features.append(self.features[i])
                    new_labels.append(self.labels[i])
            self.features = new_features
            self.labels = new_labels

        if one_hot == False:
            self.labels = np.array(self.labels)
        else:
            labels_temp = np.zeros([len(self.labels), 4])
            for i in range(len(self.labels)):
                labels_temp[i, self.labels[i]] = 1
            self.labels = labels_temp
        self.features = np.array(self.features)

    def full_data(self):

        return self.features, self.labels

    def lookup_subpart(self, seq):

        return self.dict_subpart_2_fullpart[seq]

    def lookup_timechain(self, seq):

        return self.dict_subpart_2_timechain[seq]
