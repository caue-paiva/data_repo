from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools

model = CategoricalNB()
encoder = LabelEncoder()
df = pd.read_csv("kaggle_compe\olym_treated_NoNaN_train.csv")

all_columns= ['Year']
categorical_columns = ['Sex']
best_score = -1

#chosen_columns = ["Medal", "NOC"]

for combo in list(itertools.combinations(all_columns, 1)):
   print(list(combo))
   X = df[list(combo)]
   Y = df['Games']

   print(Y.value_counts(0))

   seed = 42
   X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3 ,random_state=seed)
   X_train2 =X_train
   X_test2 = X_test


   for col in set(categorical_columns).intersection(combo):
      encoder.fit(X_train2[col])
      X_train2[col] = encoder.transform(X_train[col])
      X_test[col] = X_test[col].map(lambda s: encoder.transform([s])[0] if s in encoder.classes_ else 0)
   
   print(X_train2.shape)


   # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
   # print((X_train['Sex'] < 0).any())
   # print((Y_train < 0).any())



   model.fit(X_train2,Y_train)

   npX =  np.array(X_train2)
   npY = np.array(Y_train)
   #model.fit(npX,npY)
   # Should be False

   score = model.score(X_test, Y_test)
   if score > best_score: best_score = score
   print("Score:", score)

   print("Min:", np.min(npX, axis=0))
   #print("Max:", np.max(npY, axis=0))


   def plot_decision_boundary(X, y, model, ax):
      x_min, x_max = max(0, X[:, 0].min() - 1), X[:, 0].max() + 1
      y_min, y_max = max(0, X[:, 1].min() - 1), X[:, 1].max() + 1

      xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                           np.arange(y_min, y_max, 0.1))

      Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
      Z = Z.reshape(xx.shape)

      ax.contourf(xx, yy, Z, alpha=0.8)
      scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)

      legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
      ax.add_artist(legend1)

   Y_pred = model.predict(X_test)

   #Plotting the decision boundary
   cm = confusion_matrix(Y_test, Y_pred)
   # Plot the confusion matrix
   plt.figure(figsize=(10, 7))
   sns.set(font_scale=1.2)
   sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
   plt.xlabel('Predicted')
   plt.ylabel('Actual')
   plt.show()