# Look back train and test
import h2o
from h2o.estimators import H2ODeepLearningEstimator, H2ORandomForestEstimator, H2OGradientBoostingEstimator
from featureeng import XMLParser

from featureeng import Frame
import pandas as pd

h2o.init()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_frame = Frame(train)
test_frame = Frame(test)

train_frame.look_back(input_column='value', steps=5)
test_frame.look_back(input_column='value', steps=5)

train_frame = train_frame.get_panda_frame()
test_frame = test_frame.get_panda_frame()

train_frame = XMLParser.apply_feature_eng(pandas_frame=train_frame, xml_file='feature')
test_frame = XMLParser.apply_feature_eng(pandas_frame=test_frame, xml_file='feature')

response_column = 'is_anomaly'
training_columns = list(train_frame.columns)
training_columns.remove(response_column)

h_train = h2o.H2OFrame(train_frame)
h_train.set_names(list(train_frame.columns))
h_train[response_column] = h_train[response_column].asfactor()

h_test = h2o.H2OFrame(test_frame)
h_test.set_names(list(test_frame.columns))
h_test[response_column] = h_test[response_column].asfactor()

h_train.describe()
model = H2ODeepLearningEstimator(epochs=100, balance_classes=True, variable_importances=True)
model.train(x=training_columns, y=response_column, training_frame=h_train)

print model.model_performance(test_data=h_test)
model.varimp(use_pandas=True).to_csv('varimp.csv')

