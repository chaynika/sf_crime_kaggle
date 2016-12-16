import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

# Convert Dates column to day, month, year, daypart
sub=pd.DataFrame(train.Dates.str.split(' ').tolist(), columns = "date time".split())
date=pd.DataFrame(sub.date.str.split('-').tolist(), columns="year month day".split())
time = pd.DataFrame(sub.time.str.split(':').tolist(), columns = "hour minute second".split())
train['year'] = date['year']
train['month'] = date['month']
train['day'] = date['day'].astype(int)
train['hour'] = time['hour'].astype(int)
bins=[-1,4,8,12,16,20,23]
group_names=["daypart1","daypart2","daypart3","daypart4","daypart5","daypart6"]
train['daypart']=pd.cut(train['hour'],bins,labels=group_names)
#train['part_of_month']=pd.cut(train['day'],3,labels=["Start of month", "Mid month","End of month"])
train['part_of_month']=pd.cut(train['day'],6,labels=["1","2","3","4","5","6"])
del train['Dates']
#del train['Category']
del train['Descript']
del train['Resolution']
del train['day']
del train['hour']
del train['Address']
#### SAME FOR TEST ####

sub=pd.DataFrame(test.Dates.str.split(' ').tolist(), columns = "date time".split())
date=pd.DataFrame(sub.date.str.split('-').tolist(), columns="year month day".split())
time = pd.DataFrame(sub.time.str.split(':').tolist(), columns = "hour minute second".split())
test['year'] = date['year']
test['month'] = date['month']
test['day'] = date['day'].astype(int)
test['hour'] = time['hour'].astype(int)
bins=[-1,4,8,12,16,20,23]
group_names=["daypart1","daypart2","daypart3","daypart4","daypart5","daypart6"]
test['daypart']=pd.cut(test['hour'],bins,labels=group_names)
test['part_of_month']=pd.cut(test['day'],6,labels=["1","2","3","4","5","6"])
del test['Dates']
del test['hour']
del test['day']
del test['Address']


class_encoder=LabelEncoder()
class_encoder.fit(list(set(train.Category)))
train.Category=class_encoder.transform(train.Category)
labels=train.Category
del train['Category']

le = LabelEncoder()
le.fit(list(set(train.daypart)))
test.daypart=le.transform(test.daypart)
train.daypart=le.transform(train.daypart)

le.fit(list(set(train.part_of_month)))
train.part_of_month=le.transform(train.part_of_month)
test.part_of_month=le.transform(test.part_of_month)

le.fit(['Monday', 'Tuesday', 'Friday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday'])
train.DayOfWeek=le.transform(train.DayOfWeek)
test.DayOfWeek=le.transform(test.DayOfWeek)

le.fit(list(set(train.PdDistrict)))
train.PdDistrict=le.transform(train.PdDistrict)
test.PdDistrict=le.transform(test.PdDistrict)


#### Implementing different classifiers ####
clf = RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=2, random_state=0)
clf = clf.fit(train, labels)
scores1 = clf.score(train, labels)

### ExtraTrees has slightly better performance

clf = ExtraTreesClassifier(max_depth=None, min_samples_split=2, random_state=0)
clf = clf.fit(train, labels)
scores2 = clf.score(train, labels)

### Now plot some data to analyze

## Apply Naive Bayes
res=gnb.fit(train[['DayOfWeek', 'PdDistrict', 'X', 'Y', 'year', 'month', 'daypart', 'part_of_month']],train['Category'])\
    .predict(train[['DayOfWeek', 'PdDistrict', 'X', 'Y', 'year', 'month', 'daypart', 'part_of_month']])
a=((labels != res).sum())
accuracy=float((len(train)-a)*100/len(train))
print("Accuracy for GNB is:", accuracy)

## Implement KNN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train, labels)
ans=neigh.predict(train)
a=((ans != labels).sum())
accuracy=float((len(train)-a)*100/len(train))
print("Accuracy for KNN is:", accuracy)
print("Accuracy is seen to decrease as K increases")

## Do I need PCA?


### Try Cross validation

## Add xgboost




