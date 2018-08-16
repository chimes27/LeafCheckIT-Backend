'''
Description: Classifying image features using trained SVM
Author: J. Tejada
Version: 2.0
Date: 9/13/16
'''


import cv2
import argparse
import cPickle as pickle
import numpy as np
import os
import createfeatures as cf

from sklearn import preprocessing
#from training import SVMClassify

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Extracts features \ from each line and classifies the data')
    parser.add_argument("-img", dest="input_image", required=True, help="Input image to be classified")
    parser.add_argument("-svm", dest="svm_file", required=True,help="File containing the trained SVM model")
    parser.add_argument("-featuremap", dest="featuremap_file", required=True, help="File containing the codebook")
    return parser

class ImageClassifier():
    def __init__(self, img, svmFile, labels):
        self.encodeLabels(labels)

        imgFeat = np.array(self.getImageFeature(img), dtype=np.float32)

        self.svm = cv2.SVM()
        predictedLabel = self.predict(imgFeat)
        wordLabel = self.decodeLabels(predictedLabel)
        print(wordLabel)
        #print(predictedLabel)


    def predict(self, imgFeat):
        self.svm.load(svmFile)
        predictedLabel = self.svm.predict(imgFeat)
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
    svmFile = args.svm_file
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

    imgArgs = ImageClassifier(img, svmFile, labels)


