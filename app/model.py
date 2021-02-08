import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

#read the data
data = pd.read_csv('diabetes.csv') 
#print(data.head())

data = data.rename(columns = {'DiabetesPedigreeFunction' : 'DPF'}) # rename of column of DiabbetesPedigreeFunction
#print(df)

# replace all zero values in column to Nan values
columns = ['Glucose','BloodPressure','SkinThickness','Insulin', 'BMI']
data[columns] = data[columns].replace(to_replace = 0, value = np.nan)


# impute all NAN values using based on distribution of features
data['Glucose'].fillna(data['Glucose'].mean(), inplace = True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace = True)
data['Insulin'].fillna(data['Insulin'].median(), inplace = True)
data['BMI'].fillna(data['BMI'].median(), inplace = True)
data['SkinThickness'].fillna(data['SkinThickness'].mean(), inplace = True)
data.head()

# seperate fetures and class label
X = data.drop(['Outcome'], axis = 1)
y = data['Outcome']
print(X.shape)
print(y.shape)

# splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
print('The shape of X_train is:', X_train.shape)
print('The shape of X_test is:', X_test.shape)
print('The shape of y_train is:', y_train.shape)
print('The shape of y_test is:', y_test.shape)

# training of model
def batch_predict(clf, data):
    y_data_pred = []
    tr_loop = data.shape[0]-data.shape[0]%1000
    for i in range(0, tr_loop, 1000):
        y_data_pred.extend(clf.predict_proba(data[i:i+1000])[:,1])
    if data.shape[0]%1000 !=0:
        y_data_pred.extend(clf.predict_proba(data[tr_loop:])[:,1])
    return y_data_pred

# Training of model for batch prediction 
#from sklearn.metrics import roc_curve, auc

RFC = RandomForestClassifier(n_estimators= 100, max_depth = 5, random_state=42)
RFC.fit(X_train, y_train)

'''y_pred = batch_predict(RFC, X_test)

fpr, tpr, thresholds= roc_curve(y_test, y_pred) 
print(auc(fpr, tpr))

# confusion matrix
from sklearn.metrics import confusion_matrix
def find_best_threshold(threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    return t

def predict_with_best_t(proba, threshould):
    predictions = []
    for i in proba:
        if i>=threshould:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

best_t = find_best_threshold(thresholds, fpr, tpr)

print("Test confusion matrix")
print(confusion_matrix(y_test, predict_with_best_t(y_pred, best_t)))'''

# creating a pickle file for the clssifier
filename = 'diabetes_prediction_RFC_model.pkl'
pickle.dump(RFC, open(filename, 'wb'))
