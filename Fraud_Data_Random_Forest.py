##############################################################################
####################### R A N D O M    F O R E S T ###########################
##############################################################################

#### Importing packages and loading dataset ############

import pandas as pd
import numpy as np
import matplotlib.pyplot

fraud_data = pd.read_csv("C:\\Users\\home\Desktop\\Data Science Assignments\\Python_codes\\Random_Forest\\Fraud_check.csv")

fraud_data.head()
fraud_data.columns




#our target variable is "taxable_income" and converting it to two categories
# treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
##Converting the Taxable income variable to bucketing. 

fraud_data["income"]="<=30000"

fraud_data.loc[fraud_data["Taxable.Income"]>=30000,"income"]="Good"
fraud_data.loc[fraud_data["Taxable.Income"]<=30000,"income"]="Risky"
#assigning income column into two categories risky and good as per the Taxable>income column




##Droping the Taxable income variable
fraud_data.drop(["Taxable.Income"],axis=1,inplace=True)
#this column is no more required 

#to reduce the complexity lets change the columns names
fraud_data.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)



## Model doesnt not consider String ,So lets label the categorical columns

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in fraud_data.columns:
    if fraud_data[column_name].dtype == object:
        fraud_data[column_name] = le.fit_transform(fraud_data[column_name])
    else:
        pass
  
##Splitting the data into i/p and o/p
features = fraud_data.iloc[:,0:5]
labels = fraud_data.iloc[:,5]

## Collecting the column names

colnames = list(fraud_data.columns)
predictors = colnames[0:5]#feature variable 
target = colnames[5]# targated variable





################# Splitting the data into TRAIN and TEST ##########################

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)


############################################################################
############################ MODEL BUILDING ################################
############################################################################


from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)

model.estimators_
model.classes_
model.n_features_
model.n_classes_

model.n_outputs_

model.oob_score_  #0.7208333333333333
###72.833%

##Predictions on train data
prediction = model.predict(x_train)

##Accuracy
# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)#0.9916666666666667
##99.16%

np.mean(prediction == y_train)#0.9916666666666667
##99.16%

##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)

##Prediction on test data
pred_test = model.predict(x_test)

##Accuracy
acc_test =accuracy_score(y_test,pred_test)# 0.75
##75.00%

## In random forest we can plot a Decision tree present in Random forest
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image 
tree = model.estimators_[5]

dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

## Creating pdf and png file the selected decision tree
graph.write_pdf('fraud_data.pdf')
graph.write_png('fraud_data.png')
Image(graph.create_png())
