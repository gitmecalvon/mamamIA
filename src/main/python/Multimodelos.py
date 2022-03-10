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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import  VotingClassifier


def split(data,rnd_st):
    train,test  = train_test_split(data.iloc[:,1:-1], test_size=0.3,random_state=rnd_st)
    train=train.replace('M',1) 
    train=train.replace('B',0) 
    test=test.replace('M',1) 
    test=test.replace('B',0) 
    # test=pd.read_csv('TEST.csv',index_col=0)
    x_test=test[test.columns[1:len(test.columns)]]
    y_test=test['diagnosis']

    # train=pd.read_csv('TRAIN.csv',index_col=0)
    x_train=train[train.columns[1:len(test.columns)]]
    y_train=train['diagnosis']

    col=['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'area_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst']


    x_test=x_test[col]
    x_train=x_train[col]

    return x_train, x_test, y_train, y_test

def estandar(x_train,x_test):

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns)
    x_test=pd.DataFrame(scaler.fit_transform(x_test),columns=x_train.columns)

def logistic_regresion(x_train, x_test, y_train, y_test,result,rnd_st):

    modelLR = LogisticRegression(random_state=rnd_st) #valores por defecto
    modelLR = modelLR.fit(x_train, y_train)

    y_predLR = modelLR.predict_proba(x_test)
    y_predLR_met = modelLR.predict(x_test)


    conf_rf       = confusion_matrix(y_test, y_predLR_met)
    acc_rf        = accuracy_score(y_test, y_predLR_met)
    prec_rf       = precision_score(y_test, y_predLR_met, average="weighted")
    rec_rf        = recall_score(y_test, y_predLR_met, average="weighted")
    f1_rf         = f1_score(y_test, y_predLR_met, average="weighted")

    result=result.append([{'Random state':rnd_st,'Modelo':'Logistic Regression','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])

    return y_predLR,result

def SVM(x_train, x_test, y_train, y_test,result,rnd_st):

    modelSVM=svm.SVC(random_state=rnd_st,probability=True) #valores por defecto
    modelSVM = modelSVM.fit(x_train, y_train)

    y_predSVM = modelSVM.predict_proba(x_test)
    y_predSVM_met = modelSVM.predict(x_test)


    conf_rf       = confusion_matrix(y_test, y_predSVM_met)
    acc_rf        = accuracy_score(y_test, y_predSVM_met)
    prec_rf       = precision_score(y_test, y_predSVM_met, average="weighted")
    rec_rf        = recall_score(y_test, y_predSVM_met, average="weighted")
    f1_rf         = f1_score(y_test, y_predSVM_met, average="weighted")

    result=result.append([{'Random state':rnd_st,'Modelo':'SVM','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])

    return y_predSVM,result

def MLP(x_train, x_test, y_train, y_test,result,rnd_st):

    modelMLP=MLPClassifier(random_state=rnd_st,max_iter=400) #valores por defecto
    modelMLP = modelMLP.fit(x_train, y_train)

    y_predMLP = modelMLP.predict_proba(x_test)
    y_predMLP_met = modelMLP.predict(x_test)


    conf_rf       = confusion_matrix(y_test, y_predMLP_met)
    acc_rf        = accuracy_score(y_test, y_predMLP_met)
    prec_rf       = precision_score(y_test, y_predMLP_met, average="weighted")
    rec_rf        = recall_score(y_test, y_predMLP_met, average="weighted")
    f1_rf         = f1_score(y_test, y_predMLP_met, average="weighted")

    result=result.append([{'Random state':rnd_st,'Modelo':'MLP Classification','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])

    return y_predMLP,result

def classific(y_pred,y_test,y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN):
    cut=0.5
    # y_pred=y_pred.to_list()
    y_test=y_test.tolist()
    for i in range(len(y_pred)):
        
        if ((y_pred[i][1]>=cut)and(y_test[i]==1)):
            y_prob_TP.append(y_pred[i][1])
        elif ((y_pred[i][1]>=cut)and (y_test[i]==0)):
            y_prob_FP.append(y_pred[i][1])
        elif ((y_pred[i][1]<cut)and (y_test[i]==1)):
            y_prob_FN.append(y_pred[i][1])
        elif ((y_pred[i][1]<cut)and (y_test[i]==0)):
            y_prob_TN.append(y_pred[i][1])
    return y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN
    
def graf_prob(name,y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN):
    fig,ax=plt.subplots(2,2)
    n_bin=25
    ax[0,0].hist(y_prob_TP,color='red',align='mid',bins=n_bin)
    ax[0,0].set_title('Verdaderos positivos')
    ax[0,1].hist(y_prob_FP,color='blue',align='mid',bins=n_bin)
    ax[0,1].set_title('Falsos positivos')
    ax[1,0].hist(y_prob_FN,color='orange',align='mid',bins=n_bin)
    ax[1,0].set_title('Falsos negativos')
    ax[1,1].hist(y_prob_TN,color='green',align='mid',bins=n_bin)
    ax[1,1].set_title('Verdaderos negativos')
    fig.suptitle(name)
    plt.show()

def graf_metricas(result):
    fig1,ax1=plt.subplots(1)
    
    ax1.boxplot(result.drop(['Random state'], axis=1),labels=['Exactitud','Precisión','Recall','F-score'])  
    fig1.suptitle('Métricas')

def Bagging(data,result1):
    for model,random_state in [[svm.SVC(random_state=41,probability=True),41],[LogisticRegression(random_state=34),34],[MLPClassifier(random_state=41,max_iter=400),41]]:
        x_train, x_test, y_train, y_test=split(data,random_state)
        x_train,x_test=estandar(x_train,x_test)

        modelo=BaggingClassifier(base_estimator=model, random_state=random_state).fit(x_train, y_train)
        y_predSVM = modelo.predict_proba(x_test)
        y_predSVM_met = modelo.predict(x_test)

        conf_rf       = confusion_matrix(y_test, y_predSVM_met)
        acc_rf        = accuracy_score(y_test, y_predSVM_met)
        prec_rf       = precision_score(y_test, y_predSVM_met, average="weighted")
        rec_rf        = recall_score(y_test, y_predSVM_met, average="weighted")
        f1_rf         = f1_score(y_test, y_predSVM_met, average="weighted")

        result1=result1.append([{'Random state':random_state,'Modelo':'SVM','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])

    return result1

def AdaBoosting(data,result2):
    
    for model,random_state in [[svm.SVC(random_state=41),41],[LogisticRegression(random_state=34),34]]: #,[MLPClassifier(random_state=41,max_iter=400),41]]:
        x_train, x_test, y_train, y_test=split(data,random_state)
        x_train,x_test=estandar(x_train,x_test)

        modelo=AdaBoostClassifier(base_estimator=model, random_state=random_state,algorithm='SAMME').fit(x_train, y_train)
        y_predSVM = modelo.predict_proba(x_test)
        y_predSVM_met = modelo.predict(x_test)

        conf_rf       = confusion_matrix(y_test, y_predSVM_met)
        acc_rf        = accuracy_score(y_test, y_predSVM_met)
        prec_rf       = precision_score(y_test, y_predSVM_met, average="weighted")
        rec_rf        = recall_score(y_test, y_predSVM_met, average="weighted")
        f1_rf         = f1_score(y_test, y_predSVM_met, average="weighted")

        result2=result2.append([{'Random state':random_state,'Modelo':'SVM','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])

    return result2

def SimpleModel(data):
    
    y_prob_TP=[]
    y_prob_FP=[]
    y_prob_TN=[]
    y_prob_FN=[]
    y_predSVM=[]
    result=pd.DataFrame(columns=['Random state','Accuracy','Precision','Recall','F1 Score'])
    # for k in range(200):
    # k=34
    k=41
    x_train, x_test, y_train, y_test=split(data,k)
    x_train,x_test=estandar(x_train,x_test)

    # y_predSVM,result=SVM(x_train, x_test, y_train, y_test,result,k)
    y_predSVM,result=MLP(x_train, x_test, y_train, y_test,result,k)
    # y_predSVM,result=logistic_regresion(x_train, x_test, y_train, y_test,result,k)

    y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN=classific(y_predSVM,y_test,y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)
        
    graf_prob('MLP prediction',y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)

    # graf_metricas(result)
    maximos=pd.DataFrame(columns=['Random state','Modelo','Accuracy','Precision','Recall','F1 Score'])
    #for nn in ['Decision Tree','Random forest','Gradient Boosting','Extra tree','Logistic Regression','SVM','K neighbours mas cercanos','MLP Classification']:
    maximos=maximos.append(result.iloc[np.where(result['F1 Score']==np.max(result['F1 Score']))])
    maximos=maximos.drop_duplicates()

    print('*************')

def Voting(data):
    
    y_prob_TP=[]
    y_prob_FP=[]
    y_prob_TN=[]
    y_prob_FN=[]
    result=pd.DataFrame(columns=['Random state','Accuracy','Precision','Recall','F1 Score'])
    # for k in range(200):
    k=158
    x_train, x_test, y_train, y_test=split(data,k)
    x_train,x_test=estandar(x_train,x_test)

    modelLR = LogisticRegression(random_state=k)
    modelSVM=svm.SVC(random_state=k,probability=True)
    modelMLP=MLPClassifier(random_state=k,max_iter=400)

    modelo=VotingClassifier(estimators=[('lr', modelLR), ('svm', modelSVM), ('mlp', modelMLP)], voting='soft',weights=[0,0,1])
    modelo=modelo.fit(x_train, y_train)
    y_predSVM = modelo.predict_proba(x_test)
    y_predSVM_met = modelo.predict(x_test)

    conf_rf       = confusion_matrix(y_test, y_predSVM_met)
    acc_rf        = accuracy_score(y_test, y_predSVM_met)
    prec_rf       = precision_score(y_test, y_predSVM_met, average="weighted")
    rec_rf        = recall_score(y_test, y_predSVM_met, average="weighted")
    f1_rf         = f1_score(y_test, y_predSVM_met, average="weighted")

    # print(classification_report(y_test, y_predSVM_met))
    # print("Confusion Matrix: \n", conf_rf, '\n')
    # print("Accuracy    : ", acc_rf)
    # print("Recall      : ", prec_rf)
    # print("Precision   : ", rec_rf)
    # print("F1 Score    : ", f1_rf)

    result=result.append([{'Random state':k,'Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])

    y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN=classific(y_predSVM,y_test,y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)
        
    graf_prob('Voting',y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)

    graf_metricas(result)
    maximos=pd.DataFrame(columns=['Random state','Modelo','Accuracy','Precision','Recall','F1 Score'])
    #for nn in ['Decision Tree','Random forest','Gradient Boosting','Extra tree','Logistic Regression','SVM','K neighbours mas cercanos','MLP Classification']:
    maximos=maximos.append(result.iloc[np.where(result['F1 Score']==np.max(result['F1 Score']))])
    maximos=maximos.drop_duplicates()

    print('*************')

data=pd.read_csv("data.csv")
result1=pd.DataFrame(columns=['Random state','Modelo','Accuracy','Precision','Recall','F1 Score'])
result2=pd.DataFrame(columns=['Random state','Modelo','Accuracy','Precision','Recall','F1 Score'])

# result1=Bagging(data,result1)
# result2=AdaBoosting(data,result2)

SimpleModel(data)
#Voting(data)

print('FIN')