import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('PZ2_DesicionTree_drug200.csv')
print(df.head())
x = df.iloc[:, :-1]  
y = df.iloc[:, -1]   
x = pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = tree.DecisionTreeClassifier(criterion="entropy")
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)
plt.figure(figsize=(12, 6))  
tree.plot_tree(model, filled=True, feature_names=x.columns, class_names=np.unique(y).astype(str))
plt.title('Decision Tree Visualization')
plt.show()
