# Check individual file
from os import listdir

import h2o
from h2o.estimators import H2ODeepLearningEstimator, H2ORandomForestEstimator, H2OGradientBoostingEstimator
from os.path import isfile, join

h2o.init(max_mem_size_GB=4)

train = h2o.import_file('train.csv')

response_column = 'anomaly'
training_columns = train.col_names
training_columns.remove(response_column)

train[response_column] = train[response_column].asfactor()

# model = H2ODeepLearningEstimator()
# model = H2ORandomForestEstimator()
model = H2OGradientBoostingEstimator()
model.train(x=training_columns, y=response_column, training_frame=train)

path = 'C:/Users/Roshan/PycharmProjects/Yahoo/Webscope/A3Benchmark/'
csvfiles = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]

accuracy = []
area_under_curve = []

for file in csvfiles[-20:]:
    print file
    test = h2o.import_file(path + file)
    test[response_column] = test[response_column].asfactor()
    performance = model.model_performance(test_data=test)
    print performance
    area_under_curve.append(performance.auc())
    accuracy.append(performance.accuracy()[0])

print "Average accuracy", sum(accuracy) / float(len(accuracy))
print "Average area under curve", sum(area_under_curve) / float(len(area_under_curve))
# print model.varimp(use_pandas=True).to_csv('varimp.csv')

