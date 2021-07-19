import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2

world_data = pd.read_csv('world.csv',encoding = 'ISO-8859-1', na_values = '..')
life_data = pd.read_csv('life.csv', encoding='ISO-8859-1', na_values='..')

life_data = life_data.drop(['Year'], axis=1)
world_data = world_data.drop(['Time', 'Country Name'], axis=1)

all_data = world_data.merge(life_data,  on='Country Code')
all_data.sort_values(by='Country Code', inplace=True)

data = all_data.drop(['Country', 'Country Code', 'Life expectancy at birth (years)'], axis=1).astype(float)
target = all_data['Life expectancy at birth (years)']


X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.7, test_size=0.3, random_state=4)

imp_median = SimpleImputer(missing_values=np.nan, strategy='median').fit(X_train)
X_train = imp_median.transform(X_train)
X_test = imp_median.transform(X_test)

interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit(X_train)
X_train_pf = interaction.transform(X_train)
X_test_pf = interaction.transform(X_test)

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X_train)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('k', fontsize=25)
plt.ylabel('Distortion', fontsize=25)
plt.title('The Elbow Method showing the optimal k', fontsize=20)
plt.savefig("task2graph1.png")


kmeans = KMeans(n_clusters = 3).fit(X_train)
train_labels = kmeans.labels_
test_labels = kmeans.predict(X_test)

X_train = np.concatenate((X_train_pf, train_labels.reshape(-1,1)),axis=1)
X_test = np.concatenate((X_test_pf, test_labels.reshape(-1, 1)), axis=1)
print("Training model dimensions after implemeting feature engineering: ", X_train.shape)

selector = SelectKBest(chi2, k=4).fit(X_train, y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)
k3_accuracy_fe = np.round(accuracy_score(y_test, y_pred), decimals=3)


X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.7, test_size=0.3, random_state=4)

imp_median = SimpleImputer(missing_values=np.nan, strategy='median').fit(X_train)
X_train = imp_median.transform(X_train)
X_test = imp_median.transform(X_test)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=4).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)
k3_accuracy_pca = np.round(accuracy_score(y_test, y_pred), decimals=3)

ff_data = data[data.columns[0:4]]


X_train, X_test, y_train, y_test = train_test_split(ff_data, target, train_size=0.7, test_size=0.3, random_state=4)

imp_median = SimpleImputer(missing_values=np.nan, strategy='median').fit(X_train)
X_train = imp_median.transform(X_train)
X_test = imp_median.transform(X_test)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)
k3_accuracy_ff = np.round(accuracy_score(y_test, y_pred), decimals=3)



print("Accuracy of feature engineering:", k3_accuracy_fe)
print("Accuracy of PCA:", k3_accuracy_pca)
print("Accuracy of first four features:", k3_accuracy_ff)