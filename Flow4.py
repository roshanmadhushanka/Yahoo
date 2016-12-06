import h2o
from h2o.estimators import H2ODeepLearningEstimator, H2ORandomForestEstimator, H2OGradientBoostingEstimator

h2o.init(max_mem_size_GB=4)

train = h2o.import_file('train.csv')
test = h2o.import_file('test.csv')

response_column = 'anomaly'
training_columns = train.col_names
training_columns.remove(response_column)

train[response_column] = train[response_column].asfactor()
test[response_column] = test[response_column].asfactor()

# model = H2ODeepLearningEstimator()
model = H2ORandomForestEstimator(nfolds=10, balance_classes=True, ntrees=600, max_depth=20)
# model = H2OGradientBoostingEstimator()
model.train(x=training_columns, y=response_column, training_frame=train)
print model.model_performance(test_data=test)

# print model.varimp(use_pandas=True).to_csv('varimp.csv')

