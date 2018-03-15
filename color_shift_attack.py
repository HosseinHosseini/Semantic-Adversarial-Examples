"""
Implementation of generating semantic adversarial examples (SAEs).
Running this file will generate adversarial color-shifted images of 
CIFAR10 test data on a pretrained VGG16 model. The code outputs 
attack success rate and stores adversarial examples in a .npy file.
"""

from keras.models import load_model
from keras.utils import np_utils
from keras.datasets import cifar10

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from utils_cifar10 import *

####################################################################

def color_shift_attack(X, y, num_trials):

    N = X.shape[0] # number of samples

    # extract out images that the model misclassifies
    pred_label = np.argmax(model.predict(data_normalize(X)), axis=1)

    wrong_labels = pred_label != y.reshape(-1)
    X = X[wrong_labels == 0]
    y = y[wrong_labels == 0]

    # adv_succ_num[i]: number of adversarial samples generated after i trials
    # adv_succ_num[0] is number of clean images misclassified by model
    adv_succ_num = np.zeros((num_trials + 1, 1))
    adv_succ_num[0] = np.sum(wrong_labels)

    print('Trial ' + str(0) +
          ', Attack success rate: ' + str(np.sum(adv_succ_num) / N))

    ####################################################################

    X_adv = []  # accumulator of adversarial examples

    # Convert RGB to HSV
    X_hsv = matplotlib.colors.rgb_to_hsv(X / 255.)

    X /= 255.

    for i in range(num_trials):

        # Randomly shift Hue and Saturation components

        X_adv_hsv = np.copy(X_hsv)

        d_h = np.random.uniform(0, 1, size=(X_adv_hsv.shape[0],1))
        d_s = np.random.uniform(-1, 1, size=(X_adv_hsv.shape[0],1)) * float(i) / num_trials

        for j in range(X_adv_hsv.shape[0]):
            X_adv_hsv[j, :, :, 0] = (X_hsv[j, :, :, 0] + d_h[j]) % 1.0
            X_adv_hsv[j, :, :, 1] = np.clip(X_hsv[j, :, :, 1] + d_s[j], 0., 1.)

        X = matplotlib.colors.hsv_to_rgb(X_adv_hsv)
        X = np.clip(X, 0., 1.)

        # extract out wrongly-classified images

        pred_label = np.argmax(model.predict(data_normalize(X * 255.)), axis=1)

        wrong_labels = pred_label != y.reshape(-1)

        # store wrongly-classified images
        X_adv.append(X_hsv[wrong_labels])

        X_hsv = X_hsv[wrong_labels == 0]
        y = y[wrong_labels == 0]

        adv_succ_num[i + 1] = np.sum(wrong_labels)

        print('Trial ' + str(i+1) +
              ', Attack success rate: ' + str(np.sum(adv_succ_num)/N))

    return X_adv, adv_succ_num

####################################################################

if __name__ == '__main__':

    # load data
    X_train, y_train, X_test, y_test = cifar_data()

    # load model
    model_name = 'cifar10vgg.h5'
    model = load_model_VGG16(model_name)

    # apply attack on CIFAR10 test data

    num_trials = 1000

    color_shifted_examples, adv_succ_num = color_shift_attack(X_test, y_test, num_trials)

    np.save('color_shift_examples.npy', color_shifted_examples)
    
    # Median number of trials needed to generate adversarial examples
    median_num = np.argmin(np.abs(np.cumsum(adv_succ_num)/X_test.shape[0] - 0.5))

    print('Attack success rate: ' + str(np.sum(adv_succ_num)/X_test.shape[0]) +
          ', Median number of trials needed: ' + str(median_num))

    plt.plot(np.cumsum(adv_succ_num)/X_test.shape[0])
    plt.xlabel('Number of trials')
    plt.ylabel('Attack success rate')
    plt.show()
