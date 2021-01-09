# Flower_classification_terminal_applicaiton

Terminal application to classifiy the flower image.

# Recruitments
Python
tensorflow/tensorflow-gpu
tensorflow_hub
tensorflow_datasets
numpy
matplotlib

# Dataset 

The dataset used is the  Oxford Flowers 102 dataset from the tesnorflow datasets.

# files

In the Image_Classifier.ipynb notebook, the data is loaded and preprocessed and explored and the a pretrained model (Mobilenet) is tuned on the dataset and the best model is saved to be used in the terminal application.

prediction.py file contains the code used to predict the classes using the saved model

predict.py is the main file that call the other files and is excueted from the terminal

# usage 

To use these codess, first save the trained model
Run the followung line from the terminal: python predict.py iinput_image_paht model_path --top_k --category_names
