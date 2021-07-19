import pandas as pd
import numpy as np

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

world_data = pd.read_csv('world.csv',encoding = 'ISO-8859-1', na_values = '..')
life_data = pd.read_csv('life.csv', encoding='ISO-8859-1', na_values='..')

life_data = life_data.drop(['Year'], axis=1)
world_data = world_data.drop(['Time', 'Country Name'], axis=1)

all_data = world_data.merge(life_data,  on='Country Code')
all_data.sort_values(by='Country Code', inplace=True)

data = all_data.drop(['Country', 'Country Code', 'Life expectancy at birth (years)'], axis=1).astype(float)
target = all_data['Life expectancy at birth (years)']

X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.7, test_size=0.3, random_state=200)

med_list = np.around(np.nanmedian(X_train, axis=0), decimals=3)
imp_median = SimpleImputer(missing_values=np.nan, strategy='median').fit(X_train)
X_train = imp_median.transform(X_train)
X_test = imp_median.transform(X_test)

mean_list = np.around(np.mean(X_train, axis=0), decimals=3)
var_list = np.around(np.var(X_train, axis=0), decimals=3)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)
k3_accuracy = np.round(accuracy_score(y_test, y_pred), decimals=3)

knn = neighbors.KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)
k7_accuracy = np.round(accuracy_score(y_test, y_pred), decimals=3)



dt = DecisionTreeClassifier(random_state=200, max_depth=3)
dt.fit(X_train, y_train)

y_pred=dt.predict(X_test)
dt_accuracy = np.round(accuracy_score(y_test, y_pred), decimals=3)

features = list(data.columns)
comb_data = pd.DataFrame({'feature':features, 'median':med_list, 'mean':mean_list, 'variance':var_list})    
comb_data.to_csv('task2a.csv', index=False)

print('Accuracy of decision tree:', dt_accuracy)
print('Accuracy of k-nn (k=3):', k3_accuracy)
print('Accuracy of k-nn (k=7):', k7_accuracy)