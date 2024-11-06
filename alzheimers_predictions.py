import pandas as pd

#read csv
df = pd.read_csv('C:/Users/apsto/SpyderProjects/Data/alzheimers_prediction/alzheimers_disease_data.csv')

#split data into x and y.
y = df['Diagnosis']
x = df.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis = 1)

#clean the data
    #check for duplicates
duprow = df[df.duplicated(keep = False)]

#split Data into training and testing sets
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 42)
   
#model evaluation
from sklearn.metrics import accuracy_score as acc, classification_report as crp

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rc = RandomForestClassifier(n_estimators = 100, random_state = 42) #create new random forest model
rc.fit(x_train, y_train)

y_rc_train_pred = rc.predict(x_train)  #naming convention: axis_algorithm_typeOfSet_pred
y_rc_test_pred = rc.predict(x_test)

rc_accuracy = acc(y_test, y_rc_test_pred)
rc_clreport = crp(y_test, y_rc_test_pred)

def performance():
    print("Accuracy:", rc_accuracy)
    print("Classification Report:")
    print(rc_clreport)
    

"""work on visualizations"""





















