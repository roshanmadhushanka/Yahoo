# Remove random features and train the model

import h2o
from h2o.estimators import H2ODeepLearningEstimator, H2ORandomForestEstimator, H2OGradientBoostingEstimator

h2o.init(max_mem_size_GB=4)

train = h2o.import_file('train.csv')
test = h2o.import_file('test.csv')



response_column = 'anomaly'
train[response_column] = train[response_column].asfactor()
test[response_column] = test[response_column].asfactor()

model = H2ODeepLearningEstimator(nfolds=10, balance_classes=True, variable_importances=True)
# model = H2ORandomForestEstimator()
# model = H2OGradientBoostingEstimator()

col_list = train.col_names
col_list.remove(response_column)

for column in col_list:
    print "Removed column", column
    training_columns = train.col_names
    training_columns.remove(response_column)
    training_columns.remove(column)
    model.train(x=training_columns, y=response_column, training_frame=train)
    performance = model.model_performance(test_data=test)
    print performance
    model.varimp(use_pandas=True).to_csv(column + '_varimp.csv')
    print "------------------------------------------------------------------------------"

