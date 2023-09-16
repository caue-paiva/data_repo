from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
import numpy as np
from sklearn.decomposition import PCA

model = CategoricalNB()
encoder = LabelEncoder()
df = pd.read_csv("kaggle_compe\olym_treated_NoNaN_train.csv")

categorical_columns = ['Team','NOC','City','Sport','Event', 'Medal']



X = df.drop(columns=["Games", "Name"])
print(X.shape)
Y = df['Games']
print(Y.shape)
seed = 42
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3 ,random_state=seed)

for col in categorical_columns:
   X_train[col]= encoder.fit_transform(X_train[col])
   X_test[col]= encoder.fit_transform(X_test[col])

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model.fit(X_train,Y_train)

print(model.score(X_test, Y_test))


npX =  np.array(X_train)
npY = np.array(Y_train)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(npX)

print(X_pca)
plot_decision_regions( X_pca ,npY, clf=model)
plt.xlabel("decision boundary")
plt.show()

