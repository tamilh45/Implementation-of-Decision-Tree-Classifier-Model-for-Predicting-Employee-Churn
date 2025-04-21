# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use the criterion as entropy

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: TAMIL PAVALAN M
RegisterNumber:  212223110058
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = {
    'SatisfactionLevel': [0.8, 0.6, 0.3, 0.9, 0.5, 0.2, 0.7, 0.4],
    'LastEvaluation': [0.9, 0.8, 0.7, 0.95, 0.5, 0.3, 0.85, 0.45],
    'NumberProject': [3, 4, 2, 5, 2, 1, 3, 2],
    'AverageMonthlyHours': [200, 180, 160, 210, 150, 100, 190, 140],
    'TimeSpentCompany': [3, 4, 2, 5, 2, 1, 3, 2],
    'WorkAccident': [0, 0, 1, 0, 1, 0, 0, 1],
    'PromotionLast5Years': [0, 1, 0, 0, 1, 0, 0, 1],
    'Churn': [0, 0, 1, 0, 1, 1, 0, 1]  # Target variable
}

df = pd.DataFrame(data)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(16, 10))
plot_tree(clf, feature_names=X.columns, class_names=['Stay', 'Leave'], filled=True)
plt.title("Decision Tree for Employee Churn Prediction")
plt.show()
```


## Output:

![image](https://github.com/user-attachments/assets/d440f94a-467e-4c06-98a4-0d731f068eff)

![image](https://github.com/user-attachments/assets/c11b11e2-c35b-4b55-b246-4b4afea04b63)

![image](https://github.com/user-attachments/assets/b7b81b85-710d-4d3a-9f2c-5811d6d9401c)






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
