PHAS0077 Project
------------------

Hiya! I'm Xiaoman Ding. This is the code part for my research project for MSc in Scientific and Data Intensive Computing. My project is Blood Glucose Prediction Using Machine Learning Models, which is to use the physical examination data of Chinese people to train machine learning models and then predict blood glucose values.


File Structure
--------------
The data folder contains the dataset of this project.
train.csv:          the training set that contains medical examination data of 5642 Chinese people
test.csv:           the test set that contains medical examination data of 1000 Chinese people (without blood glucose)
answer.csv:         the target value, or the blood glucose value, for samples in the test set.
describe_train.csv: some statistical information of the training set, not very important.

For the code:
data_analysis.py:
    analyze the dataset, such as the ratio of diabetic people in the dataset, the missing ratio of every feature... If you run this file, you will get some plots about data analysis.
models.py:
    load data, do data preprocessing, build and train the models. There are functions for training Linear Regression, XGBoost, LightGBM, CatBoost, Random Forest and stacking models. If you run this file, it will start to train all the 5 single models and 1 stacking models, and output their best MSE as I stated in the Experiments and Results section of dissertation.


Dependency
----------
The programming language used is python, the modules you need to import are:
time
numpy
pandas
dateutil
sklearn
catboost
xgboost
mlxtend
matplotlib.pyplot
seaborn


Build Instructions
------------------
After installing all the required modules, you can just download the project from Github.
Run models.py if you want to see the training processes of machine learning models.
Run data_analysis.py if you want to see some plots about data analysis.