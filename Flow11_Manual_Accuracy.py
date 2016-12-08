import pandas as pd

actual = pd.read_csv('original.csv')
predict = pd.read_csv('predictions.csv')

actual = map(str, actual['value'])
predict = map(str, predict['predict'])

match = 0
mismatch = 0
for i in range(len(actual)):
    if actual[i] == predict[i]:
        match += 1
    else:
        mismatch += 1

print "Match      :", match
print "Mismatched :", mismatch
