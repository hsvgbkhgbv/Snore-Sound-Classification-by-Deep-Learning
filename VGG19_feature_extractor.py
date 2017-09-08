from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model, Sequential
import numpy as np
from snore_data_extractor import *
import os


# load pretrained VGG19 model
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# display the summary of VGG19 model
model.summary()

# set up the load path and the save path
save_folder_path = "./snore_spectrogram_10_VGG19_f1/"
load_folder_path = "/data/jw11815/snore_spectrogram_10/"

# check the existence of save path
if not os.path.exists(save_folder_path):
    os.mkdir(save_folder_path)

# set up the name of each part of dataset
targets = ["train", "devel", "test"]
timechain = False

# extract featrues from VGG19 for train, devel and test data
for t in range(len(targets)):
    print ("\n"+targets[t]+"data extarction starts!\n")
    target = snore_data_extractor(load_folder_path, one_hot=False, data_mode=targets[t], ifResize=True, resize=(224, 224), timechain=timechain, duplicate=False)
    target_features, target_labels = target.full_data()
    target_features_preprocess = preprocess_input(np.array(target_features, dtype=np.float64))
    target_fc_features = model.predict(target_features_preprocess)
    with open(save_folder_path+targets[t]+"_data.txt", "w+") as doc:
        for i in range(len(target_labels)):
            for number in target_fc_features[i]:
                doc.write(str(number))
                doc.write(",")
            doc.write("\t")
            if timechain == True:
                doc.write(str(target_labels[i])+"\t")
                doc.write(str(target.lookup_subpart(i))+"\t")
                doc.write(str(target.lookup_timechain(i))+"\n")
            else:
                doc.write(str(target_labels[i])+"\n")
    print ("\n"+targets[t]+"data extraction ends!\n")
