import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

random.seed(2)
rnd_st=2
#----------------
# CONTENIDO     |
#----------------
	# Muestra variables más importantes
	# Métricas de todos los modelos
	# Posibilidad de seleccionar solo un número de variables

#***********************************************************************
#----------------
# LECTURA DATOS |
#----------------
test=pd.read_csv('TEST.csv',index_col=0)
x_test=test[test.columns[1:len(test.columns)]]
y_test=test['diagnosis']

train=pd.read_csv('TRAIN.csv',index_col=0)
x_train=train[test.columns[1:len(test.columns)]]
y_train=train['diagnosis']

# En caso de querer seleccionar otras columnas, descomentar la que sea y comentar el resto

# col=['radius_mean', 'perimeter_mean', 'area_mean','concave points_mean','area_se','radius_worst','perimeter_worst', 'area_worst', 'concavity_worst', 'concave points_worst',]

# col=['radius_mean', 'texture_mean', 'perimeter_mean',
#        'area_mean', 'compactness_mean', 'concavity_mean',
#        'concave points_mean', 'area_se', 'radius_worst', 'texture_worst',
#        'perimeter_worst', 'area_worst', 'smoothness_worst',
#        'compactness_worst', 'concavity_worst', 'concave points_worst']

# Descomentar si se quiere seleccionar unas columnas

# x_test=x_test[col]
# x_train=x_train[col]


#***********************************************************************
#---------------------------
# ESTANDARIZACIÓN DE DATOS |
#---------------------------

scaler = StandardScaler()
scaler.fit(x_train)
x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns)
x_test=pd.DataFrame(scaler.fit_transform(x_test),columns=x_train.columns)

result=pd.DataFrame(columns=['Modelo','Accuracy','Precision','Recall','F1 Score'])


#***********************************************************************
#----------------------
# REGRESION LOGISTICA |
#----------------------
print('***************************************')
print('| LOGISTIC REGRESION |')
print('----------------------')

modelLR = LogisticRegression(random_state=rnd_st) #valores por defecto
modelLR = modelLR.fit(x_train, y_train)

importance = abs(modelLR.coef_[0])
for i,v in enumerate(importance):
	print('Feature: %s, Importance: %.5f' % (x_train.columns[i],v))

plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))],x_train.columns,rotation='vertical')
plt.title('Regresión logistica')
plt.tight_layout()
plt.show()
y_predLR = modelLR.predict(x_test)


conf_rf       = confusion_matrix(y_test, y_predLR)
acc_rf        = accuracy_score(y_test, y_predLR)
prec_rf       = precision_score(y_test, y_predLR, average="weighted")
rec_rf        = recall_score(y_test, y_predLR, average="weighted")
f1_rf         = f1_score(y_test, y_predLR, average="weighted")

result=result.append([{'Modelo':'Logistic Regression','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


print(classification_report(y_test, y_predLR))
print("Confusion Matrix: \n", conf_rf, '\n')
print("Accuracy    : ", acc_rf)
print("Recall      : ", prec_rf)
print("Precision   : ", rec_rf)
print("F1 Score    : ", f1_rf)




#***********************************************************************
#----------------
# DECISION TREE |
#----------------
print('***************************************')
print('| ÁRBOL DE DECISION |')
print('---------------------')

modelDT1 = DecisionTreeClassifier(random_state=rnd_st) #valores por defecto
modelDT1 = modelDT1.fit(x_train, y_train)

plt.figure(figsize=(16,6),tight_layout=1.5)
tree.plot_tree(modelDT1,fontsize=8)
plt.show()

importance = modelDT1.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %s, Importance: %.5f' % (x_train.columns[i],v))

plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))],x_train.columns,rotation='vertical')
plt.title('Árbol de decisión')
plt.tight_layout()
plt.show()
y_predDT1 = modelDT1.predict(x_test)


conf_rf       = confusion_matrix(y_test, y_predDT1)
acc_rf        = accuracy_score(y_test, y_predDT1)
prec_rf       = precision_score(y_test, y_predDT1, average="weighted")
rec_rf        = recall_score(y_test, y_predDT1, average="weighted")
f1_rf         = f1_score(y_test, y_predDT1, average="weighted")

result=result.append([{'Modelo':'Decision Tree','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


print(classification_report(y_test, y_predDT1))
print("Confusion Matrix: \n", conf_rf, '\n')
print("Accuracy    : ", acc_rf)
print("Recall      : ", prec_rf)
print("Precision   : ", rec_rf)
print("F1 Score    : ", f1_rf)


#***********************************************************************
#----------------
# RANDOM FOREST |
#----------------
print('***************************************')
print('| RANDOM FOREST |')
print('---------------------')

modelRF = RandomForestClassifier(random_state=rnd_st) #valores por defecto
modelRF = modelRF.fit(x_train, y_train)


importance = modelRF.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %s, Importance: %.5f' % (x_train.columns[i],v))
plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))],x_train.columns,rotation='vertical')
plt.title('Random forest')
plt.tight_layout()
plt.show()
y_predRF = modelRF.predict(x_test)


conf_rf       = confusion_matrix(y_test, y_predRF)
acc_rf        = accuracy_score(y_test, y_predRF)
prec_rf       = precision_score(y_test, y_predRF, average="weighted")
rec_rf        = recall_score(y_test, y_predRF, average="weighted")
f1_rf         = f1_score(y_test, y_predRF, average="weighted")

result=result.append([{'Modelo':'Random forest','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


print(classification_report(y_test, y_predRF))
print("Confusion Matrix: \n", conf_rf, '\n')
print("Accuracy    : ", acc_rf)
print("Recall      : ", prec_rf)
print("Precision   : ", rec_rf)
print("F1 Score    : ", f1_rf)


#***********************************************************************
#----------------
# SVM           |
#----------------
print('***************************************')
print('| SVM |')
print('---------------------')

modelSVM=svm.SVC(random_state=rnd_st) #valores por defecto
modelSVM = modelSVM.fit(x_train, y_train)


y_predSVM = modelSVM.predict(x_test)


conf_rf       = confusion_matrix(y_test, y_predSVM)
acc_rf        = accuracy_score(y_test, y_predSVM)
prec_rf       = precision_score(y_test, y_predSVM, average="weighted")
rec_rf        = recall_score(y_test, y_predSVM, average="weighted")
f1_rf         = f1_score(y_test, y_predSVM, average="weighted")

result=result.append([{'Modelo':'SVM','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


print(classification_report(y_test, y_predSVM))
print("Confusion Matrix: \n", conf_rf, '\n')
print("Accuracy    : ", acc_rf)
print("Recall      : ", prec_rf)
print("Precision   : ", rec_rf)
print("F1 Score    : ", f1_rf)


#***********************************************************************
#--------------------
# Gradient Boosting |
#--------------------
print('***************************************')
print('| Gradient Boosting |')
print('---------------------')

modelGB=GradientBoostingClassifier(random_state=rnd_st) #valores por defecto
modelGB = modelGB.fit(x_train, y_train)

y_predGB = modelGB.predict(x_test)


conf_rf       = confusion_matrix(y_test, y_predGB)
acc_rf        = accuracy_score(y_test, y_predGB)
prec_rf       = precision_score(y_test, y_predGB, average="weighted")
rec_rf        = recall_score(y_test, y_predGB, average="weighted")
f1_rf         = f1_score(y_test, y_predGB, average="weighted")

result=result.append([{'Modelo':'Gradient Boosting','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


print(classification_report(y_test, y_predGB))
print("Confusion Matrix: \n", conf_rf, '\n')
print("Accuracy    : ", acc_rf)
print("Recall      : ", prec_rf)
print("Precision   : ", rec_rf)
print("F1 Score    : ", f1_rf)


#***********************************************************************
#----------------
# EXTRA TREE    |
#----------------
print('***************************************')
print('| EXTRA TREE |')
print('---------------------')

modelET=ExtraTreesClassifier(random_state=rnd_st) #valores por defecto
modelET = modelET.fit(x_train, y_train)


importance = modelET.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %s, Importance: %.5f' % (x_train.columns[i],v))
plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))],x_train.columns,rotation='vertical')
plt.title('Extra tree')
plt.tight_layout()
plt.show()
y_predET = modelET.predict(x_test)


conf_rf       = confusion_matrix(y_test, y_predET)
acc_rf        = accuracy_score(y_test, y_predET)
prec_rf       = precision_score(y_test, y_predET, average="weighted")
rec_rf        = recall_score(y_test, y_predET, average="weighted")
f1_rf         = f1_score(y_test, y_predET, average="weighted")

result=result.append([{'Modelo':'Extra tree','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


print(classification_report(y_test, y_predET))
print("Confusion Matrix: \n", conf_rf, '\n')
print("Accuracy    : ", acc_rf)
print("Recall      : ", prec_rf)
print("Precision   : ", rec_rf)
print("F1 Score    : ", f1_rf)


#***********************************************************************
#----------------
# KNN    |
#----------------
print('***************************************')
print('| KNN |')
print('---------------------')

modelKNN=KNeighborsClassifier()#random_state=rnd_st) #valores por defecto
modelKNN = modelKNN.fit(x_train, y_train)

y_predKNN = modelKNN.predict(x_test)


conf_rf       = confusion_matrix(y_test, y_predKNN)
acc_rf        = accuracy_score(y_test, y_predKNN)
prec_rf       = precision_score(y_test, y_predKNN, average="weighted")
rec_rf        = recall_score(y_test, y_predKNN, average="weighted")
f1_rf         = f1_score(y_test, y_predKNN, average="weighted")

result=result.append([{'Modelo':'K neighbours mas cercanos','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


print(classification_report(y_test, y_predKNN))
print("Confusion Matrix: \n", conf_rf, '\n')
print("Accuracy    : ", acc_rf)
print("Recall      : ", prec_rf)
print("Precision   : ", rec_rf)
print("F1 Score    : ", f1_rf)


#***********************************************************************
#----------------
# MLP Classifier    |
#----------------
print('***************************************')
print('| MLP Classification |')
print('---------------------')

modelMLP=MLPClassifier(random_state=rnd_st) #valores por defecto
modelMLP = modelMLP.fit(x_train, y_train)



y_predMLP = modelMLP.predict(x_test)


conf_rf       = confusion_matrix(y_test, y_predMLP)
acc_rf        = accuracy_score(y_test, y_predMLP)
prec_rf       = precision_score(y_test, y_predMLP, average="weighted")
rec_rf        = recall_score(y_test, y_predMLP, average="weighted")
f1_rf         = f1_score(y_test, y_predMLP, average="weighted")

result=result.append([{'Modelo':'MLP Classification','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


print(classification_report(y_test, y_predMLP))
print("Confusion Matrix: \n", conf_rf, '\n')
print("Accuracy    : ", acc_rf)
print("Recall      : ", prec_rf)
print("Precision   : ", rec_rf)
print("F1 Score    : ", f1_rf)

#***********************************************************************
#----------------
# COMPARAMOS MODELOS    |
#----------------
result.index=result['Modelo']
result=result.drop(['Modelo'], axis=1)
result.plot.bar()
plt.tight_layout()
plt.show()

print('FIN')