from AlexNet import *
import numpy as np
from snore_data_extractor import *
import os


x_input = tf.placeholder(tf.float32, (None,227, 227, 3))

# load pretrained AlexNet model
output_layer = AlexNet(x_input, "f1")

# set up the load path and the save path
save_folder_path = "./snore_spectrogram_10_AlexNet_f1/"
load_folder_path = "/data/jw11815/snore_spectrogram_10/"

# check the existence of save path
if not os.path.exists(save_folder_path):
    os.mkdir(save_folder_path)

# set up the name of each part of dataset
targets = ["train", "devel", "test"]
timechain = False

# extract featrues from AlexNet for train, devel and test data
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for t in range(len(targets)):
        print ("\n"+targets[t]+"data extarction starts!\n")
        target = snore_data_extractor(load_folder_path, one_hot=False, data_mode=targets[t], ifResize=True, resize=(227, 227), timechain=timechain, duplicate=False)
        target_features, target_labels = target.full_data()
        target_features = np.array(target_features)
        target_features_preprocess = preprocess_input(np.array(target_features, dtype=np.float64))
        target_fc_features = sess.run(output_layer, feed_dict = {x_input:target_features_preprocess})
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
