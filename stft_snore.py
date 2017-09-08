import numpy as np
from librosa.core import stft
import os
from scipy import signal, misc
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa


save_file = "../snore_spectrogram_10/"

if not os.path.exists(save_file):
    os.mkdir(save_file)

if not os.path.exists(save_file+"train"):
    os.mkdir(save_file+"train")

if not os.path.exists(save_file+"devel"):
    os.mkdir(save_file+"devel")

if not os.path.exists(save_file+"test"):
    os.mkdir(save_file+"test")

i_train = 0
i_devel = 0
i_test = 0

counter_list = []
for signal in os.listdir("../Snore_dist/wav"):

    sig_vec = signal.split('_')
    sig_name = sig_vec[0]
    sig_number = int(sig_vec[1].split(".")[0])
    print ("This is the current signal to be processed: "+signal)
    sigbag = wavfile.read("../Snore_dist/wav/"+signal)
    sig = sigbag[1]
    #sig = sig / np.amax(abs(sig))
    #counter_list.append(np.floor(sig.shape[0]))


    sig = sig[:44000]

    if sig_name == "train":
        i_train = sig_number
    elif sig_name == "devel":
        i_devel = sig_number
    elif sig_name == "test":
        i_test = sig_number


    if sig.shape[0] >= 44000:
        pass
    else:
        while sig.shape[0] < 44000:
            sig = np.concatenate((sig, sig[0:44000-sig.shape[0]]))
            #sig = np.concatenate((sig, np.zeros([44000-sig.shape[0], ])))
    ms = np.square(np.abs(stft(y=sig, n_fft=350, hop_length=175)))
    #ms = np.abs(stft(y=sig, n_fft=350, hop_length=175))
    #ms = librosa.amplitude_to_db(stft(y=sig, n_fft=256, hop_length=128))

    # save the resultant gradient matrix as image
    if sig_name == "train":
        plt.imsave(save_file+"train/"+str(i_train)+'.png',ms,cmap="viridis")
    elif sig_name == "devel":
        plt.imsave(save_file+"devel/"+str(i_devel)+'.png',ms,cmap="viridis")
    elif sig_name == "test":
        plt.imsave(save_file+"test/"+str(i_test)+'.png',ms,cmap="viridis")

    if sig_name == "train":
        print ("This is train "+str(i_train)+" finished!")
    elif sig_name == "devel":
        print ("This is devel "+str(i_devel)+" finished!")
    elif sig_name == "test":
        print ("This is test "+str(i_test)+" finished!")


#plt.hist(counter_list)
#plt.show()
