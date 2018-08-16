'''
Testing SVM Cross Validation Using OPENCV SVM

'''

import cv2
import argparse
import cPickle as pickle
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import timeit




def build_arg_parser():
    parser = argparse.ArgumentParser(description="Creates features for given images")
    parser.add_argument("-featuremap", dest='feature_map_file', required=True,
                        help="Locate generated feature map format: -featuremap <model_path/featuremap_filename.pkl>")
    #parser.add_argument("-svmfile", dest="svm_file", required=False,
                        #help="Specify the path the svm file to be saved: -svmfile <model_path/svm_filename.pkl>")
    return parser


class SVMClassify(object):
    def __init__(self, features, label, C=1, gamma=0.5, ): ### gamma = 0.5
        '''
        result(4- fold cross-validation of kernels):
        C=1; gamma = 0.5
            Error result:
                RBF =  83.33%
                Linear = 50.00
                poly = 50
        '''

        self.params = dict(kernel_type=cv2.SVM_RBF,
                           svm_type=cv2.SVM_C_SVC,
                           C=C,
                           gamma=gamma)

        
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label)
        leLabels = np.array(self.le.transform(label), dtype=np.float32)

        features_train, features_test, labels_train, labels_test = train_test_split(features, leLabels, test_size=0.2, random_state=0)


        result = self.training(features_train, labels_train)
        #result = self.training(features, leLabels)

        create_report(features_train)
        #print(labels_train)
        self.crossValidation(result, features_test, labels_test)

        #self.crossValidation(result, features, leLabels)
        self.model = cv2.SVM()

    def training(self, features, label):
        features = np.array(features, dtype=np.float32)
        self.model = cv2.SVM()
        #self.model.getParams(self.params1)
        self.model.train(features, label, params=self.params1)
        #self.model.train(features,label, ????)
        #self.model.train(features, label, params=self.params2)
        self.model.save("svm.xml")

    def predict(self, features):
        labels_nums = self.model.predict(features)
        labels_words = self.le.inverse_transform([int(x) for x in labels_nums])
        return labels_words

    def crossValidation(self, model, features_test, labels_test):
        features_test = np.array(features_test, dtype=np.float32)
        predictionResult = []
        for item in features_test:
            res = self.model.predict(item)
            predictionResult.append(res)

        accuracy = (labels_test == predictionResult).mean()
        error = (labels_test != predictionResult).mean()

        print("Manual Accuracy Score: %.2f %%" % (accuracy * 100))
        print("Error: %.2f %%" % (error * 100))

        '''
        confusion = np.zeros((3, 3), np.int32)
        for i, j in zip(labels_test, predictionResult):
            confusion[i, j] += 1
        print 'confusion matrix:'
        print confusion
        '''
        return

def create_report(mat):
    with open('test.txt', 'w') as file:
        for item in mat:
            file.write("{}\n".format(item))
    print("Report Created")
    return


if __name__ == '__main__':
    start_time = timeit.default_timer()
    args = build_arg_parser().parse_args()
    feature_map_file = args.feature_map_file

    #svmfileName = args.svm_file

    with open(feature_map_file, 'r') as f:
        feature_map = pickle.load(f)

    features = []
    labels = []
    for item in feature_map:
        features.append(item['features'])
        labels.append(item['label'])

    svm = SVMClassify(features, labels)
    print("The SVM File has been saved!")
    elapsed = timeit.default_timer() - start_time
    print("Time Elapsed: " + str(elapsed) + " seconds")
