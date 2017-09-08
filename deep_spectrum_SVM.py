from sklearn.metrics import confusion_matrix, recall_score
from sklearn.svm import SVC
from deep_model_feature_reader import *
import numpy as np
import os
import pickle


def singleSVC(target_features, target_labels, kernel, training, C, mode, s=False):

    if training == True:
        model = SVC(C=C, kernel=kernel)
        model.fit(target_features, target_labels)
        s = pickle.dumps(model)
    else:
        model = pickle.loads(s)

    predict_labels = model.predict(target_features)
    target_fit = recall_score(target_labels, predict_labels, average="macro")

    print ("This is the "+mode+" data recall score: ", target_fit)
    print (confusion_matrix(target_labels, predict_labels))
    print ("\n")

    if training == True:
        return target_fit, s
    else:
        return target_fit


def train_mode(load_file_path, save_file_path, file_name, kernel, C_range):

    train = deep_model_feature_reader(load_file_path, data_mode="train", timechainParameter=False)
    train_features, train_labels = train.full_data()
    devel = deep_model_feature_reader(load_file_path, data_mode="devel", timechainParameter=False)
    devel_features, devel_labels = devel.full_data()

    with open(save_file_path+file_name, "w+") as doc:
        print ("start tuning ...")
        for iteration in range(C_range.shape[0]):
            print ("\nIteration: ", iteration)
            print ("C is: ", C_range[iteration])
            train_fit, s = singleSVC(target_features=train_features, target_labels=train_labels, kernel=kernel, training=True, C=C_range[iteration], mode="train", s=False)
            devel_fit = singleSVC(target_features=devel_features, target_labels=devel_labels, kernel=kernel, training=False, C=C_range[iteration], mode="devel", s=s)
            doc.write(str(C_range[iteration])+"\t"+str(train_fit)+"\t"+str(devel_fit)+"\n")


def test_mode(load_file_path, kernel, C):

    train = deep_model_feature_reader(load_file_path, data_mode="train", timechainParameter=False)
    devel = deep_model_feature_reader(load_file_path, data_mode="devel", timechainParameter=False)
    test = deep_model_feature_reader(load_file_path, data_mode="test", timechainParameter=False)
    train_features, train_labels = train.full_data()
    devel_features, devel_labels = devel.full_data()
    test_features, test_labels = test.full_data()
    train_features = np.concatenate((train_features, devel_features), axis=0)
    train_labels = np.concatenate((train_labels, devel_labels), axis=0)

    train_fit, s = singleSVC(target_features=train_features, target_labels=train_labels, kernel=kernel, training=True, C=C, mode="train&devel", s=False)
    devel_fit = singleSVC(target_features=test_features, target_labels=test_labels, kernel=kernel, training=False, C=C, mode="test", s=s)


if __name__ == "__main__":

    # the path of loading features extracted from deep model
    load_file_path = "./snore_spectrogram_10_VGG19_f2/"
    # the path of recordig training procedure
    save_file_path = "./record/"
    # the file name of the recording training procedure
    file_name = "SVM_snore_spectrogram_10_VGG19_f2"
    # the mode of "train" or "test"
    mode = "train"

    # setups for training and testing
    ############################################################################
    # the rough range of hyperparamters for training, if in test mode please annotate it
    C_range=np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    # the specific range of hyperparamters for training, if in test mode pelase annotate it
    C_range = np.linspace(0.001, 0.1, 100)
    # the hyperparamter for testing, if in train mode please annotate it
    C = 0.043
    # set up the kernel of SVM
    kernel = "linear"

    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)

    # process of training and testing
    ############################################################################
    if mode == "train":
        train_mode(load_file_path, save_file_path, file_name, kernel, C_range)
    elif mode == "test":
        test_mode(load_file_path, kernel, C)
