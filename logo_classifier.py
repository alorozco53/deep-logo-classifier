#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import argparse

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, cdist
from ipywidgets import interact, interact_manual
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

model = VGG16(weights='imagenet')

# Load dataset
dataset = pd.read_pickle('dataset.pkl')
enc = LabelEncoder()
enc.fit_transform(dataset.get_values().ravel())

# Load knn models
cnn = joblib.load('cnn-train-knn.pkl')
hist = joblib.load('hist-train-knn.pkl')

def build_conv_model():
    return VGG16(weights='imagenet')

def get_conv_features(train_path, model):
    features = {}
    for i, img in enumerate(os.listdir(train_path)):
        if img.endswith('jpg') or img.endswith('jpeg') or img.endswith('png'):
            im = image.load_img(os.path.join(train_path, img),
                                 target_size=(224, 224))
            x = image.img_to_array(im)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features[img] = model.predict(x).squeeze()
        if i % 100 == 0:
            print('Step:', i)
    features = pd.DataFrame(features)
    return features

def build_knn(k, data, serialize=False, prefix='train-', jobs=1):
    """
    Build and train K-NN
    """
    knn = NearestNeighbors(n_neighbors=k, n_jobs=jobs)
    X = data.T
    knn.fit(X)
    print(knn)
    if serialize:
        print('Serializing model in:' '{}knn.pkl'.format(prefix))
        joblib.dump(knn, '{}knn.pkl'.format(prefix))
    return knn

def extract_color_histogram(train_path):
    hists = {}
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    bins = (8, 8, 8)
    for i, img in enumerate(os.listdir(train_path)):
        if img.endswith('jpg') or img.endswith('png') or img.endswith('jpeg'):
            im = cv2.imread(os.path.join(train_path, img))
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                                [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            hists[img] = hist.flatten()
            if i % 100 == 0:
                print('Step:', i)
    return pd.DataFrame(hists)

def test_sample(test, knn):
    return knn.kneighbors(test)

def knn_cnn(image_path):
    features = []
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features.append(model.predict(x).squeeze())
    features = pd.DataFrame(features)
    return test_sample(features, cnn)

def knn_hist(image_path):
    hists = []
    bins = (8, 8, 8)
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                     [0, 180, 0, 256, 0, 256])
    cv2.normalize(h, h)
    hists.append(h.flatten())
    hists = pd.DataFrame(hists)
    return test_sample(hists, hist)

def cnn_score(distances, alpha=0.8):
    return (1.0/alpha) * distances

def hist_score(distances, beta=0.2):
    return (1.0/beta) * distances

def k_best(k, image_path):
    """
    Gets the k nearest neghbors of the image given, according to the trained model.
    The value 'k' must not exceed the model's 'trained' parameter.
    """

    # get 2*k images
    distances_cnn, indices_cnn = knn_cnn(image_path)
    distances_h, indices_h = knn_hist(image_path)

    # score results
    cnn_scores = cnn_score(distances_cnn)
    hist_scores = hist_score(distances_h)

    # concatenate the results
    conc = np.concatenate((cnn_scores, hist_scores), axis=1)
    conc_indices = np.concatenate((indices_cnn, indices_h), axis=1)

    # compute k closest images
    kb = []
    for j, score_array in enumerate(conc):
        best_indices = np.argpartition(score_array, k)
        kb.append([conc_indices[j][i] for i in best_indices[:k]])
    kb = np.array(kb)
    return kb, enc.inverse_transform(kb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', required=True,
	                help='path to input dataset')
    parser.add_argument('-k', type=int, default=1,
	                help='# of nearest neighbors for classification')
    parser.add_argument('-j', type=int, default=-1,
	                help='# of jobs for k-NN distance (-1 uses all available cores)')
    parser.add_argument('-s', action='store_false',
                        help='only-store-dataset mode')
    args = parser.parse_args()

    if args.s:
        print('Building CNN model...')
        model = build_conv_model()
        print('Extracting features using a CNN...')
        features = get_conv_features(args.d, model)
        print('Extracting color histograms...')
        hists = extract_color_histogram(args.d)
        print('Training KNN model for CNN features...')
        build_knn(args.k, features, serialize=True, prefix='cnn-train-', jobs=args.j)
        print('Training KNN model for histogram features...')
        build_knn(args.k, hists, serialize=True, prefix='hist-train-', jobs=args.j)

    print('Storing dataset in \"dataset.pkl\"')
    files = [os.path.join(args.d, f) for f in os.listdir(args.d)
             if f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png')]
    pd.DataFrame(files).to_pickle('dataset.pkl')
