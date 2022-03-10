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

# random.seed(2)
# rnd_st=2
#----------------
# TAREAS |
#----------------
	#--> Seleccionar modelos finalistas
	#--> Valorar eficacia reduciendo variables
	#--> Valorar eficacia con varios modelos --> Reduciendo variables
	#--> Optimización hiperparametros con modelo seleccionado

	#--> Desarollar documento explicación

	#--> Plantear estructura streamlit
	#--> Desarrollar aplicación stramlit

	#--> Implementar más datos si es posible

#***********************************************************************
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
    n_bin=100
    ax[0,0].hist(y_prob_TP,color='red',align='left',bins=n_bin)
    ax[0,0].set_title('Verdaderos positivos')
    ax[0,1].hist(y_prob_FP,color='blue',align='left',bins=n_bin)
    ax[0,1].set_title('Falsos positivos')
    ax[1,0].hist(y_prob_FN,color='orange',align='left',bins=n_bin)
    ax[1,0].set_title('Falsos negativos')
    ax[1,1].hist(y_prob_TN,color='green',align='left',bins=n_bin)
    ax[1,1].set_title('Verdaderos negativos')
    fig.suptitle(name)
    plt.show()


def graf_metricas(metricas):
    names=['Decision tree','Random forest','Gradient boosting','Extre tree','Regresión logística','SVM','KNN','Perceptrón multicapa']
    fig1,ax1=plt.subplots(2)
    fig2,ax2=plt.subplots(2)
    ax1[0].boxplot([result.iloc[np.where(result['Modelo']=='Decision Tree')]['Accuracy'],
    result.iloc[np.where(result['Modelo']=='Random forest')]['Accuracy'],
    result.iloc[np.where(result['Modelo']=='Gradient Boosting')]['Accuracy'],
    result.iloc[np.where(result['Modelo']=='Extra tree')]['Accuracy'],
    result.iloc[np.where(result['Modelo']=='Logistic Regression')]['Accuracy'],
    result.iloc[np.where(result['Modelo']=='SVM')]['Accuracy'],
    result.iloc[np.where(result['Modelo']=='K neighbours mas cercanos')]['Accuracy'],
    result.iloc[np.where(result['Modelo']=='MLP Classification')]['Accuracy']],labels=names)
    ax1[0].set_title('Exactitud')
    
    ax1[1].boxplot([result.iloc[np.where(result['Modelo']=='Decision Tree')]['Precision'],
    result.iloc[np.where(result['Modelo']=='Random forest')]['Precision'],
    result.iloc[np.where(result['Modelo']=='Gradient Boosting')]['Precision'],
    result.iloc[np.where(result['Modelo']=='Extra tree')]['Precision'],
    result.iloc[np.where(result['Modelo']=='Logistic Regression')]['Precision'],
    result.iloc[np.where(result['Modelo']=='SVM')]['Precision'],
    result.iloc[np.where(result['Modelo']=='K neighbours mas cercanos')]['Precision'],
    result.iloc[np.where(result['Modelo']=='MLP Classification')]['Precision']],labels=names)
    ax1[1].set_title('Precisión')
    fig1.suptitle('Métricas')


    ax2[0].boxplot([result.iloc[np.where(result['Modelo']=='Decision Tree')]['Recall'],
    result.iloc[np.where(result['Modelo']=='Random forest')]['Recall'],
    result.iloc[np.where(result['Modelo']=='Gradient Boosting')]['Recall'],
    result.iloc[np.where(result['Modelo']=='Extra tree')]['Recall'],
    result.iloc[np.where(result['Modelo']=='Logistic Regression')]['Recall'],
    result.iloc[np.where(result['Modelo']=='SVM')]['Recall'],
    result.iloc[np.where(result['Modelo']=='K neighbours mas cercanos')]['Recall'],
    result.iloc[np.where(result['Modelo']=='MLP Classification')]['Recall']],labels=names)
    ax2[0].set_title('Recall')
    ax2[1].boxplot([result.iloc[np.where(result['Modelo']=='Decision Tree')]['F1 Score'],
    result.iloc[np.where(result['Modelo']=='Random forest')]['F1 Score'],
    result.iloc[np.where(result['Modelo']=='Gradient Boosting')]['F1 Score'],
    result.iloc[np.where(result['Modelo']=='Extra tree')]['F1 Score'],
    result.iloc[np.where(result['Modelo']=='Logistic Regression')]['F1 Score'],
    result.iloc[np.where(result['Modelo']=='SVM')]['F1 Score'],
    result.iloc[np.where(result['Modelo']=='K neighbours mas cercanos')]['F1 Score'],
    result.iloc[np.where(result['Modelo']=='MLP Classification')]['F1 Score']],labels=names)
    ax2[1].set_title('F-Score')
    fig2.suptitle('Métricas')

    #----------------
    # LECTURA DATOS |
    #----------------
def split(data):
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


    #***********************************************************************
    #---------------------------
    # ESTANDARIZACIÓN DE DATOS |
    #---------------------------
def estandar(x_train,x_test):

    #x_train.to_csv ("TRAIN.csv")
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns)
    x_test=pd.DataFrame(scaler.fit_transform(x_test),columns=x_train.columns)
    return x_train,x_test



    #***********************************************************************
    #----------------
    # DECISION TREE |
    #----------------
def Decision_tree(x_train, x_test, y_train, y_test,result):
    # print('***************************************')
    # print('| ÁRBOL DE DECISION |')
    # print('---------------------')

    modelDT1 = DecisionTreeClassifier(random_state=rnd_st) #valores por defecto
    modelDT1 = modelDT1.fit(x_train, y_train)

    # plt.figure(figsize=(16,6),tight_layout=1.5)
    # tree.plot_tree(modelDT1,fontsize=8)
    # plt.show()

    # importance = modelDT1.feature_importances_
    # for i,v in enumerate(importance):
    # 	print('Feature: %s, Importance: %.5f' % (x_train.columns[i],v))

    # plt.bar([x for x in range(len(importance))], importance)
    # plt.xticks([x for x in range(len(importance))],x_train.columns,rotation='vertical')
    # plt.title('Árbol de decisión')
    # plt.tight_layout()
    # plt.show()
    y_predDT1 = modelDT1.predict_proba(x_test)
    y_predDT1_met = modelDT1.predict(x_test)
    
    conf_rf       = confusion_matrix(y_test, y_predDT1_met)
    acc_rf        = accuracy_score(y_test, y_predDT1_met)
    prec_rf       = precision_score(y_test, y_predDT1_met, average="weighted")
    rec_rf        = recall_score(y_test, y_predDT1_met, average="weighted")
    f1_rf         = f1_score(y_test, y_predDT1_met, average="weighted")

    result=result.append([{'Random state':rnd_st,'Modelo':'Decision Tree','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


    # print(classification_report(y_test, y_predDT1))
    # print("Confusion Matrix: \n", conf_rf, '\n')
    # print("Accuracy    : ", acc_rf)
    # print("Recall      : ", prec_rf)
    # print("Precision   : ", rec_rf)
    # print("F1 Score    : ", f1_rf)
    return y_predDT1,result

    #***********************************************************************
    #----------------
    # RANDOM FOREST |
    #----------------
def random_forest(x_train, x_test, y_train, y_test,result):
    # print('***************************************')
    # print('| RANDOM FOREST |')
    # print('---------------------')

    modelRF = RandomForestClassifier(random_state=rnd_st) #valores por defecto
    modelRF = modelRF.fit(x_train, y_train)


    # importance = modelRF.feature_importances_
    # for i,v in enumerate(importance):
    # 	print('Feature: %s, Importance: %.5f' % (x_train.columns[i],v))
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.xticks([x for x in range(len(importance))],x_train.columns,rotation='vertical')
    # plt.title('Random forest')
    # plt.tight_layout()
    # plt.show()
    y_predRF = modelRF.predict_proba(x_test)
    y_predRF_met = modelRF.predict(x_test)


    conf_rf       = confusion_matrix(y_test, y_predRF_met)
    acc_rf        = accuracy_score(y_test, y_predRF_met)
    prec_rf       = precision_score(y_test, y_predRF_met, average="weighted")
    rec_rf        = recall_score(y_test, y_predRF_met, average="weighted")
    f1_rf         = f1_score(y_test, y_predRF_met, average="weighted")

    result=result.append([{'Random state':rnd_st,'Modelo':'Random forest','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


    # print(classification_report(y_test, y_predRF))
    # print("Confusion Matrix: \n", conf_rf, '\n')
    # print("Accuracy    : ", acc_rf)
    # print("Recall      : ", prec_rf)
    # print("Precision   : ", rec_rf)
    # print("F1 Score    : ", f1_rf)

    return y_predRF,result

    #***********************************************************************
    #--------------------
    # Gradient Boosting |
    #--------------------
def Gradient_boosting(x_train, x_test, y_train, y_test,result):
    # print('***************************************')
    # print('| Gradient Boosting |')
    # print('---------------------')

    modelGB=GradientBoostingClassifier(random_state=rnd_st) #valores por defecto
    modelGB = modelGB.fit(x_train, y_train)


    # importance = modelGB.feature_importances_
    # for i,v in enumerate(importance):
    # 	print('Feature: %s, Importance: %.5f' % (x_train.columns[i],v))
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.xticks([x for x in range(len(importance))],x_train.columns,rotation='vertical')
    # plt.title('Gradient Boosting')
    # plt.tight_layout()
    # plt.show()
    y_predGB = modelGB.predict_proba(x_test)
    y_predGB_met = modelGB.predict(x_test)


    conf_rf       = confusion_matrix(y_test, y_predGB_met)
    acc_rf        = accuracy_score(y_test, y_predGB_met)
    prec_rf       = precision_score(y_test, y_predGB_met, average="weighted")
    rec_rf        = recall_score(y_test, y_predGB_met, average="weighted")
    f1_rf         = f1_score(y_test, y_predGB_met, average="weighted")

    result=result.append([{'Random state':rnd_st,'Modelo':'Gradient Boosting','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


    # print(classification_report(y_test, y_predGB))
    # print("Confusion Matrix: \n", conf_rf, '\n')
    # print("Accuracy    : ", acc_rf)
    # print("Recall      : ", prec_rf)
    # print("Precision   : ", rec_rf)
    # print("F1 Score    : ", f1_rf)
    
    return y_predGB,result

#***********************************************************************
    #----------------
    # EXTRA TREE    |
    #----------------
def extra_tree(x_train, x_test, y_train, y_test,result):
    
    # print('***************************************')
    # print('| EXTRA TREE |')
    # print('---------------------')

    modelET=ExtraTreesClassifier(random_state=rnd_st) #valores por defecto
    modelET = modelET.fit(x_train, y_train)


    # importance = modelET.feature_importances_
    # for i,v in enumerate(importance):
    # 	print('Feature: %s, Importance: %.5f' % (x_train.columns[i],v))
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.xticks([x for x in range(len(importance))],x_train.columns,rotation='vertical')
    # plt.title('Extra tree')
    # plt.tight_layout()
    # plt.show()
    y_predET = modelET.predict_proba(x_test)
    y_predET_met = modelET.predict(x_test)

    conf_rf       = confusion_matrix(y_test, y_predET_met)
    acc_rf        = accuracy_score(y_test, y_predET_met)
    prec_rf       = precision_score(y_test, y_predET_met, average="weighted")
    rec_rf        = recall_score(y_test, y_predET_met, average="weighted")
    f1_rf         = f1_score(y_test, y_predET_met, average="weighted")

    result=result.append([{'Random state':rnd_st,'Modelo':'Extra tree','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])


    # print(classification_report(y_test, y_predET))
    # print("Confusion Matrix: \n", conf_rf, '\n')
    # print("Accuracy    : ", acc_rf)
    # print("Recall      : ", prec_rf)
    # print("Precision   : ", rec_rf)
    # print("F1 Score    : ", f1_rf)

    return y_predET,result    

    #***********************************************************************
    #----------------------
    # REGRESION LOGISTICA |
    #----------------------
def logistic_regresion(x_train, x_test, y_train, y_test,result):

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

    #***********************************************************************
    #----------------
    # SVM           |
    #----------------
def SVM(x_train, x_test, y_train, y_test,result):

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

    #***********************************************************************
    #----------------
    # KNN    |
    #----------------
def KNN(x_train, x_test, y_train, y_test,result):

    modelKNN=KNeighborsClassifier()#random_state=rnd_st) #valores por defecto
    modelKNN = modelKNN.fit(x_train, y_train)

    y_predKNN = modelKNN.predict_proba(x_test)
    y_predKNN_met = modelKNN.predict(x_test)

    conf_rf       = confusion_matrix(y_test, y_predKNN_met)
    acc_rf        = accuracy_score(y_test, y_predKNN_met)
    prec_rf       = precision_score(y_test, y_predKNN_met, average="weighted")
    rec_rf        = recall_score(y_test, y_predKNN_met, average="weighted")
    f1_rf         = f1_score(y_test, y_predKNN_met, average="weighted")

    result=result.append([{'Random state':rnd_st,'Modelo':'K neighbours mas cercanos','Accuracy':acc_rf,'Recall':rec_rf,'Precision':prec_rf,'F1 Score':f1_rf}])

    return y_predKNN,result

    #***********************************************************************
    #----------------
    # MLP Classifier    |
    #----------------

def MLP(x_train, x_test, y_train, y_test,result):
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

#***********************************************************************
    #----------------
    # COMPARAMOS MODELOS    |
    #----------------
    
    result.index=result['Modelo']
    result.drop(['Modelo'], axis=1)
    # result.plot.bar()
    plt.plot(result['Random state'],result['Accuracy'],label='Exactitud')
    plt.plot(result['Random state'],result['Precision'],label='Precisión')
    plt.plot(result['Random state'],result['Recall'],label='Recall')
    plt.plot(result['Random state'],result['F1 Score'],label='F-score')
    plt.legend()
    plt.tight_layout()
    plt.show()

models=['Decision tree','Random forest','Gradient boosting','Extre tree','Regresión logística','Super vector machine','K vecinos más próximos','Perceptrón multicapa']

result=pd.DataFrame(columns=['Random state','Modelo','Accuracy','Precision','Recall','F1 Score'])
for modelo in models:
    y_prob_TP=[]
    y_prob_FP=[]
    y_prob_TN=[]
    y_prob_FN=[]
    
    data=pd.read_csv("data.csv")
    for k in range(200):
        random.seed(k)
        rnd_st=k
        x_train, x_test, y_train, y_test=split(data)
        x_train,x_test=estandar(x_train,x_test)
        if modelo==models[0]:
            y_predDT1,result=Decision_tree(x_train, x_test, y_train, y_test,result)
        elif modelo==models[1]:
            y_predDT1,result=random_forest(x_train, x_test, y_train, y_test,result)
        elif modelo==models[2]:
            y_predDT1,result=Gradient_boosting(x_train, x_test, y_train, y_test,result)
        elif modelo==models[3]:
            y_predDT1,result=extra_tree(x_train, x_test, y_train, y_test,result)
        elif modelo==models[4]:
            y_predDT1,result=logistic_regresion(x_train, x_test, y_train, y_test,result)
        elif modelo==models[5]:
            y_predDT1,result=SVM(x_train, x_test, y_train, y_test,result)
        elif modelo==models[6]:
            y_predDT1,result=KNN(x_train, x_test, y_train, y_test,result)
        elif modelo==models[7]:
            y_predDT1,result=MLP(x_train, x_test, y_train, y_test,result)


        y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN=classific(y_predDT1,y_test,y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)
        
    if modelo==models[0]:
        graf_prob('Decision tree',y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)
    elif modelo==models[1]:
        graf_prob('Random forest',y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)
    elif modelo==models[2]:
        graf_prob('Gradient boosting',y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)
    elif modelo==models[3]:
        graf_prob('Extre tree',y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)
    elif modelo==models[4]:
        graf_prob('Regresión logística',y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)
    elif modelo==models[5]:
        graf_prob('Super vector machine',y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)
    elif modelo==models[6]:
        graf_prob('K vecinos más próximos',y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)
    elif modelo==models[7]:
        graf_prob('Perceptrón multicapa',y_prob_TP,y_prob_FP,y_prob_TN,y_prob_FN)

    #compare_model(result)
graf_metricas(result)
maximos=pd.DataFrame(columns=['Random state','Modelo','Accuracy','Precision','Recall','F1 Score'])
for nn in ['Decision Tree','Random forest','Gradient Boosting','Extra tree','Logistic Regression','SVM','K neighbours mas cercanos','MLP Classification']:
    maximos=maximos.append(result.iloc[np.where(result['F1 Score']==np.max(result.iloc[np.where(result['Modelo']==nn)]['F1 Score']))])
maximos=maximos.drop_duplicates()

print('FIN')