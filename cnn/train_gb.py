import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/home/andriy/diploma/CNN')
from utils import count_progress
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier


os.system('clear')
df = pd.read_pickle('gradient_boosting_data.pkl')

# print(df, len(df))
# arr1 = ['filename', 'accent', 'sex']
# arr2 = ['mfccs0', 'mfccs1', 'mfccs2', 'mfccs3', 'mfccs4', 'mfccs5', 'mfccs6', 'mfccs7', 'mfccs8', 'mfccs9', 'mfccs10', 'mfccs11', 'mfccs12', 'mfccs13', 'mfccs14', 'mfccs15', 'mfccs16', 'mfccs17', 'mfccs18', 'mfccs19']
# new_df = pd.DataFrame(columns=arr1 + arr2)
# for i, row in df.iterrows():
#     d = dict()
#     d['filename'] = row['filename']
#     d['accent'] = row['accent']
#     d['sex'] = row['sex']
#     j = 0
#     for mfccs in row['mfccs']:
#         d[arr2[j]] = np.mean(mfccs.T)
#         j += 1
#     new_df = new_df.append(d, ignore_index=True)
#     count_progress(i, len(df), 'dataframe expansion')
# print(new_df)
# new_df.to_pickle('gradient_boosting_data.pkl')
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

gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train, y_train)

print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))

gb_clf.fit(X_train, y_train)
predictions = gb_clf.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))