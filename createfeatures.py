'''
Description: Extracting features from image
Author: J. Tejada
Version: 2.0
Date: 9/13/16
'''

import os
import cv2
import numpy as np
import cPickle as pickle
import argparse
import timeit

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Creates features for given images")
    parser.add_argument("-s", dest="cls", nargs="+", action="append", required="True", help="format: -s <label_name> <image_path>")
    parser.add_argument("-codebook", dest='codebook_file',required=True,help="format: -codebook <code_file_name> <model_path>" )
    parser.add_argument("-featuremap", dest='feature_map_file',required=True, help="format: -featuremap <feature_map_file_name> <model_path>")
    return parser

class PrepareInputs(object):
    def load_input_map(self,label, input_dir):
        input_map=[]
        if not os.path.isdir(input_dir):
            raise IOError("The " + input_dir + "  image directory does not exist! ")
        for root, dirs, files in os.walk(input_dir):
            for filename in (x for x in files if x.endswith('.jpg') or x.endswith('.JPG') or x.endswith('.png') or x.endswith('.PNG')):
                input_map.append({'label' : label, 'image' :  os.path.join(root, filename)})

        return input_map

class FeatureMapExtraction(object):
    def getCentroid(self, input_map):
        codebook=[]
        featureMap=[]
        codebookFvs={}
        finalFeatureMap=[]
        buff=[]
        for item in input_map:
            featureMapFvs = {}
            cur_label=item['label']
            img = cv2.imread(item['image'])
            img_resize = imageResize(img)
            colorFeat = ColorExtract().getMeanStd(img_resize)
            siftKps, siftDesc = FeatureExtraction().MainExtractor(img_resize)
            clusteredFeat = FeatureCluster().kmeansFeature(siftDesc)

            codebookFvs = colorFeat + clusteredFeat
            codebook.append(codebookFvs)

            featureMapFvs['features'] = colorFeat + clusteredFeat
            featureMapFvs['label'] = cur_label

            print "Feature extracted for: " + item['image']
            if featureMap is not None:
                featureMap.append(featureMapFvs)

        #print(kps_all)
        #reps=[]
        #reps.extend(featureKmeans)
        create_report(featureMap)
        return codebook, featureMap

def imageResize(input_image):
    new_size=150
    h, w = input_image.shape[0], input_image.shape[1]
    ds_factor = new_size / float(h)

    if w < h:
        ds_factor = new_size / float(w)

    new_size = (int(w * ds_factor), int(h * ds_factor))
    newImage = cv2.resize(input_image, new_size)
    return newImage


class FeatureExtraction(object):
    def MainExtractor(self,img):
        featureKeyPoints = DenseExtract().detect(img)
        featureKeyPoints, featureVectors = SiftExtract().compute(img,featureKeyPoints)
        return featureKeyPoints, featureVectors

class DenseExtract(object):
    def __init__(self, step_size=20, feature_scale=40, img_bound=4):

        self.detector = cv2.FeatureDetector_create("Dense")
        self.detector.setInt("initXyStep", step_size)
        self.detector.setInt("initFeatureScale", feature_scale)
        self.detector.setInt("initImgBound", img_bound)


    def detect(self, img):
        '''
        densepoints = self.detector.detect(img, None)
        img2 = cv2.drawKeypoints(img, densepoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Dense', img2)
        cv2.waitKey()
        '''
        return self.detector.detect(img)


class SiftExtract(object):
    def compute(self, img, featureKeyPoints):
        if img is None:
            print "Not a valid image"
            raise TypeError

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, des = cv2.SIFT().compute(gray_image, featureKeyPoints)
        return kps, des

class ColorExtract(object):
    def getMeanStd(self, img):

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img3 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img4 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


        mean, std = cv2.meanStdDev(img)
        meanLab, stdLab = cv2.meanStdDev(img2)
        meanYCrCb, stdYCrCb = cv2.meanStdDev(img3)
        meanHSV, stdHSV = cv2.meanStdDev(img4)

        stat = np.append(mean,std).flatten()
        stat2 = np.append(meanLab,stdLab).flatten()
        stat3 = np.append(meanYCrCb,stdYCrCb).flatten()
        stat4 = np.append(meanHSV, stdHSV).flatten()
        #stat3 = meanYCrCb.flatten()

        colors = []
        for item in stat:
            colors.append(item)
        for item in stat2:
            colors.append(item)
        for item in stat3:
            colors.append(item)
        for item in stat4:
            colors.append(item)
        return colors
        '''
        #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img3 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img4 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mean, std = cv2.meanStdDev(img)



        #meanLab, stdLab = cv2.meanStdDev(img2)
        meanYCrCb, stdYCrCb = cv2.meanStdDev(img3)
        meanHSV, stdHSV = cv2.meanStdDev(img4)

        stat = np.append(mean,std).flatten()
        #stat2 = np.append(meanLab,stdLab).flatten()
        stat3 = np.append(meanYCrCb, stdYCrCb).flatten()
        stat4 = np.append(meanHSV, stdHSV).flatten()

        #stat = mean.flatten()
        #stat2 = meanLab.flatten()
        #stat3 = meanYCrCb.flatten()

        colors = []
        for item in stat:
            colors.append(item)
        #for item in stat2:
        #    colors.append(item)
        for item in stat3:
            colors.append(item)
        for item in stat4:
            colors.append(item)
        return colors
        '''
class FeatureCluster(object):
    def kmeansFeature(self,des):   ####input descriptor from Sift
        fvs = np.array(des)

        Z = fvs.reshape((-1,1))
        Z = np.float32(Z)
        flag = cv2.KMEANS_RANDOM_CENTERS
        clusters = 8 
        term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, clusters, term_crit, 10, flag)

        centroid=[]
        for item in center:
            centroid.append(item[0])
        return centroid

def create_report(mat):
    with open('newfeat.txt', 'w') as file:
        for item in mat:
            file.write("{}\n".format(item))
    print("Report Created")
    return



if __name__=='__main__':
    start_time = timeit.default_timer()
    args = build_arg_parser().parse_args()
    input_map=[]
    for cls in args.cls:
        label = cls[0]
        ##get individual input data with respective labels
        input_map += PrepareInputs().load_input_map(label,cls[1])


    print "===== Generating and calculating features ====="
    codebook, featureMap = FeatureMapExtraction().getCentroid(input_map)

    print "===== Building codebook ====="
    if args.codebook_file:
        with open(args.codebook_file, 'w') as f:
             pickle.dump((codebook), f)


    print "===== Building featureMap for Training ====="
    if args.feature_map_file:
        with open(args.feature_map_file, 'w') as f:
             pickle.dump((featureMap), f)

    print("Codebook and featureMap is already generated!")

    elapsed = timeit.default_timer() - start_time

    print("Time Elapsed: " + str(elapsed) + " seconds")
