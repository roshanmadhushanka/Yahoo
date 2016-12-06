# Look back train and test
import h2o
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2ORandomForestEstimator

from featureeng import Frame
import pandas as pd

h2o.init(max)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_frame = Frame(train)
test_frame = Frame(test)

train_frame.look_back(input_column='value', steps=12)
test_frame.look_back(input_column='value', steps=12)

response_column = 'is_anomaly'
training_columns = train_frame.get_column_names()
training_columns.remove(response_column)

#model = H2ODeepLearningEstimator(nfolds=10, epochs=100, balance_classes=True, variable_importances=True)

h_train = h2o.H2OFrame(train_frame.get_panda_frame())
h_train.set_names(list(train.columns))
h_train[response_column] = h_train[response_column].asfactor()

h_test = h2o.H2OFrame(test_frame.get_panda_frame())
h_test.set_names(list(test.columns))
h_test[response_column] = h_test[response_column].asfactor()

# model = H2ORandomForestEstimator(nfolds=10, balance_classes=True, max_depth=20, ntrees=100)
h_train.describe()
model = H2ODeepLearningEstimator(nfolds=10, activation='maxout', distribution='multinomial', epochs=100, balance_classes=True, variable_importances=True)
model.train(x=training_columns, y=response_column, training_frame=h_train)

print model.model_performance(test_data=h_test)
model.varimp(use_pandas=True).to_csv('varimp.csv')

