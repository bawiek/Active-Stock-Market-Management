import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer 
from sklearn.externals import joblib

# Import Training and Test set
path='C:/Users/batzp/Documents/Projects/FinData/Data_prep/'
NASDAQ = pd.read_csv(path+"NASDAQ.csv", sep=',')  
NASDAQ_2017_X = pd.read_csv(path+"X_Nasdaq_2017.csv", sep=',') 
NASDAQ_2017_Y = pd.read_csv(path+"Y_Nasdaq_2017.csv", sep=',')

# Data from WRDS and process in another file
##Fondamentals: 
    # Yield quarter
    # marge de profit
    # P/E
    # Total assets
    # Total liabilities
    # B
    # P/B
    # Nb shares
    # % shares issued
    # % shares repurchased

## Technical:
    # Price variation
    # Momentum 3 months
    # Momentum 6 months
    # Median quarter
    # Beat median


# RF for the explainability
def RandomForest(X_train, y_train, X_test, y_test, name):
    RF_model = RandomForestClassifier(n_estimators=20, criterion='entropy',random_state=5)
    RF_model.fit(X_train, y_train)
    y_pred = RF_model.predict(X_test)
    joblib.dump(RF_model, name)
    cm = confusion_matrix(y_test, y_pred)
    print(RF_model.feature_importances_)
    return cm


def GradientBoosting(X_train, y_train, X_test, y_test, name):
    GB_model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, random_state=5)
    GB_model.fit(X_train, y_train)
    y_pred = GB_model.predict(X_test)
    joblib.dump(GB_model, name)
    cm = confusion_matrix(y_test, y_pred)
    return cm

def SVC(X_train, y_train, X_test, y_test, name):
    SVC_model = LinearSVC(random_state=5)
    SVC_model.fit(X_train, y_train)
    y_pred = SVC_model.predict(X_test)
    joblib.dump(SVC_model, name)
    cm = confusion_matrix(y_test, y_pred)
    return cm


# Seperate var and response variables
X_Nasdaq = NASDAQ.iloc[:,0:-1]
Y_Nasdaq = NASDAQ.iloc[:,-1]
Y_Nasdaq= Y_Nasdaq.astype('float')

NASDAQ_2017_Y= NASDAQ_2017_Y.iloc[:,-1]


# Imputation
imp = SimpleImputer(missing_values= np.nan, strategy='mean')
imp = imp.fit(X_Nasdaq)
X_Nasdaq = imp.transform(X_Nasdaq)

imp = SimpleImputer(missing_values= np.nan, strategy='mean')
imp = imp.fit(NASDAQ_2017_X)
X_Nasdaq_2017 = imp.transform(NASDAQ_2017_X)



# Train test plit 
X_Nasdaq_train, X_Nasdaq_test, Y_Nasdaq_train, Y_Nasdaq_test = train_test_split(X_Nasdaq,Y_Nasdaq,test_size=0.20, random_state=0)

# Training results------------------------------------------------------------
## RANDOM FOREST
cm_Nasdaq = RandomForest(X_Nasdaq_train, Y_Nasdaq_train, X_Nasdaq_test, Y_Nasdaq_test,"NASDAQMODEL_RF.pkl")
print("precision: ")
print((cm_Nasdaq[0][0] + cm_Nasdaq[1][1]) / (cm_Nasdaq[0][0] + cm_Nasdaq[0][1] + cm_Nasdaq[1][0] + cm_Nasdaq[1][1]))
print("beating median precision: ")
print(cm_Nasdaq[1][1] / (cm_Nasdaq[0][1] + cm_Nasdaq[1][1]))
print("not beating median precision: ")
print(cm_Nasdaq[0][0] / (cm_Nasdaq[0][0] + cm_Nasdaq[1][0]))
## Gradient Boosting
cm_Nasdaq = GradientBoosting(X_Nasdaq_train, Y_Nasdaq_train, X_Nasdaq_test, Y_Nasdaq_test,"NASDAQMODEL_GB.pkl")
print("precision: ")
print((cm_Nasdaq[0][0] + cm_Nasdaq[1][1]) / (cm_Nasdaq[0][0] + cm_Nasdaq[0][1] + cm_Nasdaq[1][0] + cm_Nasdaq[1][1]))
print("beating median precision: ")
print(cm_Nasdaq[1][1] / (cm_Nasdaq[0][1] + cm_Nasdaq[1][1]))
print("not beating median precision: ")
print(cm_Nasdaq[0][0] / (cm_Nasdaq[0][0] + cm_Nasdaq[1][0]))
## Linear Support Vector Classification.
cm_Nasdaq = SVC(X_Nasdaq_train, Y_Nasdaq_train, X_Nasdaq_test, Y_Nasdaq_test,"NASDAQMODEL_SCV.pkl") #doesn't converge
print("precision: ")
print((cm_Nasdaq[0][0] + cm_Nasdaq[1][1]) / (cm_Nasdaq[0][0] + cm_Nasdaq[0][1] + cm_Nasdaq[1][0] + cm_Nasdaq[1][1]))
print("beating median precision: ")
print(cm_Nasdaq[1][1] / (cm_Nasdaq[0][1] + cm_Nasdaq[1][1]))
print("not beating median precision: ")
print(cm_Nasdaq[0][0] / (cm_Nasdaq[0][0] + cm_Nasdaq[1][0]))



# Test results ----------------------------------------------------------------
## RANDOM FOREST
RF_NASDAQ = joblib.load('NASDAQMODEL_RF.pkl')
NASDAQ_pred = RF_NASDAQ.predict(X_Nasdaq_2017)
cm_Nasdaq_2017 = confusion_matrix(NASDAQ_2017_Y, NASDAQ_pred)
print("precision: ")
print((cm_Nasdaq_2017[0][0] + cm_Nasdaq_2017[1][1]) / (cm_Nasdaq_2017[0][0] + cm_Nasdaq_2017[0][1] + cm_Nasdaq_2017[1][0] + cm_Nasdaq_2017[1][1]))
print("beating median precision: ")
print(cm_Nasdaq_2017[1][1] / (cm_Nasdaq_2017[0][1] + cm_Nasdaq_2017[1][1]))
print("not beating median precision: ")
print(cm_Nasdaq_2017[0][0] / (cm_Nasdaq_2017[0][0] + cm_Nasdaq_2017[1][0]))

## Gradient Boosting
GB_NASDAQ = joblib.load('NASDAQMODEL_GB.pkl')
NASDAQ_pred = GB_NASDAQ.predict(X_Nasdaq_2017)
cm_Nasdaq_2017 = confusion_matrix(NASDAQ_2017_Y, NASDAQ_pred)
print("precision: ")
print((cm_Nasdaq_2017[0][0] + cm_Nasdaq_2017[1][1]) / (cm_Nasdaq_2017[0][0] + cm_Nasdaq_2017[0][1] + cm_Nasdaq_2017[1][0] + cm_Nasdaq_2017[1][1]))
print("beating median precision: ")
print(cm_Nasdaq_2017[1][1] / (cm_Nasdaq_2017[0][1] + cm_Nasdaq_2017[1][1]))
print("not beating median precision: ")
print(cm_Nasdaq_2017[0][0] / (cm_Nasdaq_2017[0][0] + cm_Nasdaq_2017[1][0]))



## Linear Support Vector Classification.
RF_NASDAQ = joblib.load('NASDAQMODEL_SCV.pkl')
NASDAQ_pred = RF_NASDAQ.predict(X_Nasdaq_2017)
cm_Nasdaq_2017 = confusion_matrix(NASDAQ_2017_Y, NASDAQ_pred)
print("precision: ")
print((cm_Nasdaq_2017[0][0] + cm_Nasdaq_2017[1][1]) / (cm_Nasdaq_2017[0][0] + cm_Nasdaq_2017[0][1] + cm_Nasdaq_2017[1][0] + cm_Nasdaq_2017[1][1]))
print("beating median precision: ")
print(cm_Nasdaq_2017[1][1] / (cm_Nasdaq_2017[0][1] + cm_Nasdaq_2017[1][1]))
print("not beating median precision: ")
print(cm_Nasdaq_2017[0][0] / (cm_Nasdaq_2017[0][0] + cm_Nasdaq_2017[1][0]))