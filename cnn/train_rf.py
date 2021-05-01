import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/home/andriy/diploma/CNN')
from utils import count_progress
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

os.system('clear')
df = pd.read_pickle('gradient_boosting_data.pkl')
print(df.head())

df_accent_mfccs = df.drop(['filename', 'sex'], axis=1)


train, other = train_test_split(df_accent_mfccs, test_size=0.2, random_state=42, shuffle=True)
dev, test = train_test_split(other, test_size=0.5, random_state=21, shuffle=True)
print(len(train), len(dev), len(test))

X_test = test.drop('accent', axis=1) # setting up test set without 'ground truth'
y_train = train['accent'] # setting up y_train
X_train = train.drop('accent', axis=1)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
  
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  
    test_size=0.30, random_state=12)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

predictions = rf.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))