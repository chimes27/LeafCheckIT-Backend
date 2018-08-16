'''
Description: Classifying image features using trained SVM
Author: J. Tejada
Version: 2.0
Date: 11/2/17
'''


import cv2
import argparse
import cPickle as pickle
import numpy as np
import os
import createfeatures as cf

from sklearn import preprocessing
#from training import SVMClassify
import rftraining as rf

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Extracts features \ from each line and classifies the data')
    parser.add_argument("-img", dest="input_image", required=True, help="Input image to be classified")
    #parser.add_argument("-classifier", dest="rd_file", required=True,help="File containing the trained Random Forest trained model")
    parser.add_argument("-featuremap", dest="featuremap_file", required=True, help="File containing the codebook")
    return parser

class ImageClassifier():
    def __init__(self, img, labels):
        self.encodeLabels(labels)
        imgFeat = np.array(self.getImageFeature(img), dtype=np.float32)

        #self.model = crf.model
        #predictedLabel = self.predict(imgFeat)
        #wordLabel = self.decodeLabels(predictedLabel)
        #print(wordLabel)
        predictedLabel = rf.Classify().predict(imgFeat)
        print(predictedLabel)


    def predict(self, imgFeat):
        predictedLabel = self.model.predict(imgFeat)
        return predictedLabel

    def getImageFeature(self,img):
        imgArgs=[]
        resizedImg = cf.imageResize(img)
        colorFeat = cf.ColorExtract().getMeanStd(resizedImg)
        siftKps, siftDesc = cf.FeatureExtraction().MainExtractor(resizedImg)
        clusteredFeat = cf.FeatureCluster().kmeansFeature(siftDesc)

        imgFvs = colorFeat + clusteredFeat
        imgArgs.append(imgFvs)
        return imgArgs

    def encodeLabels(self,labels):
        self.le = preprocessing.LabelEncoder()
        return self.le.fit(labels)

    def decodeLabels(self,predictedLabel):
        return self.le.inverse_transform(int(predictedLabel))



if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    #svmFile = args.rd_file
    featuremap = args.featuremap_file

    with open(featuremap, 'r') as f:
        feature_map = pickle.load(f)

    labels=[]
    for item in feature_map:
        labels.append(item['label'])

    input_img = args.input_image
    if not os.path.isfile(input_img):
        raise IOError("The " + input_img + "  image directory does not exist! ")
    else:
        img = cv2.imread(args.input_image)

    #imgArgs = ImageClassifier(img, svmFile, labels)
    imgArgs = ImageClassifier(img, labels)

