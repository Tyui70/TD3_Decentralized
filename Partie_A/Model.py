import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Ces deux lignes de commandes permettent d'être sûr que le fichier iris.csv est dans le même que Model.py
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv('iris.csv')

X = df.drop('variety', axis=1)
y = df['variety']

# Vous pouvez changer le "test_size" par 0.9, pour éviter d'avoir 1 partout
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
classification_report_output = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report_output)

with open('model_setosa.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
