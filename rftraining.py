'''
Training AI using Random Forest Algorithm

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
    parser.add_argument("-file", dest="rt_file", required=False,
                        help="Specify the path the svm file to be saved: -RandForestFile <model_path/filename.pkl>")
    return parser


class Classify(object):
    def __init__(self, features, label, C=1, gamma=0.5, ): ### gamma = 0.5
        self.params = dict(max_depth=100,
                           min_sample_count=1,
                           use_surrogates=False,
                           max_categories=5,
                           calc_var_importance=False,
                           nactive_vars=0,
                           max_num_of_tree_in_the_forest=100,
                           term_crit=(cv2.TERM_CRITERIA_MAX_ITER,100,1))
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label)
        leLabels = np.array(self.le.transform(label), dtype=np.float32)
        features_train, features_test, labels_train, labels_test = train_test_split(features, leLabels, test_size=0.2, random_state=0)
        result = self.trainingForest(features_train, labels_train)
        create_report(features_train)
        self.crossValidation(result, features_test, labels_test)
        self.model = cv2.RTrees()

    def trainingForest(self, features, label):
        features = np.array(features, dtype=np.float32)
        self.model = cv2.RTrees()
        self.model.train(features, cv2.CV_ROW_SAMPLE, label, params=self.params)

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

    forest = Classify(features, labels)


    print("The Random Forest trained classifier has been saved!")
    elapsed = timeit.default_timer() - start_time
    print("Time Elapsed: " + str(elapsed) + " seconds")
