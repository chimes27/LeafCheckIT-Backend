# LeafCheckIT-Backend
This repository is the backend module of the LeafCheckIT application.
To use this code, you need to install the following application:
1. Python 2.7
2. OpenCV 2.4
3. Numpy
4. Scipy 0.18

The precompiled version of OpenCV and Numpy are available in the installer folder of this repository. Unfortunately, the Scipy could not be uploaded here at Github due file size constraints. Message me for a copy of the file or email me at: jonilyn.tejada@gmail.com.

Steps in using this program:
1. Run the createfeatures.py using the command prompt (windows) or terminal for Linux users:
  python createfeatures.py -s "labelNameOrCategoryName" path_to_sample_images  -codebook path_for_storing_the_model_and_filename.pkl -featuremap path_for_storing_the_featuremap_and_filename.pkl 
  
You can add as many categories as you want provided that the -s will be declared before the label name and image path.

2. For classifier training, use the following python file:
ANNTraining.py - for artificial neural network
RForestTraining.py - Random Forest
training.py - SVM training

To train, type the following in the command prompt:
python training.py -featuremap path_to_saved_featuremap.pkl

3. Access the classifier through the following files:
RForestclassification.py - Random forest
classification.py - SVM

To use the file, type the following to the command prompt:
For SVM:
python classification.py -img provide_img_path_and_filename.jpg path_to_saved_featuremap.pkl

For random forest:
python RForestclassification.py -img provide_img_path_and_filename.jpg path_to_saved_featuremap.pkl
