import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Iteración con varias semillas train test tb cambia--> comprobar que no varían los resultados (o no en exceso)

#----------------
# LECTURA DATOS |
#----------------

data=pd.read_csv('data.csv')
data=data.iloc[:,1:-1]
print(data.info())

#******************************************************************************************************
#---------------------------
# OBTENCIÓN DE VALORES NAN |
#---------------------------

print('valores NaN:')
print(str(data.isna().sum()))
print('------------')


#******************************************************************************************************
#----------------------------------------------
# PASO DE VARIABLE CUANTITATIVA A CUALITATIVA |
#----------------------------------------------

data_c=data.copy()
for i in range(np.shape(data_c)[0]): # look for replace function
    if data_c['diagnosis'][i]=='M':
        data_c['diagnosis'][i]=np.float64(1)
    else:
        data_c['diagnosis'][i]=np.float64(0)
data_c['diagnosis']=data_c['diagnosis'].astype(float)
sns.heatmap(data_c.corr())

correl=pd.DataFrame(columns=['Var','corr'])
for i in range(np.shape(data.corr())[0]-1):
       correl=correl.append({'Var':data.columns[1+i],'corr':data_c.corr()[data.columns[0]][data.columns[i+1]]})
orden=correl.sort_values('corr',ascending=False)

#******************************************************************************************************
#--------------------------------
# DESCRIPCIÓN ESTADÍSTICA DATOS |
#--------------------------------

k=7
max=4
desc=data.describe()
for i in range(0,max):
    print(data[data.columns[k*i+1:k+k*i+1]].describe())
print(data[data.columns[max*k:32]].describe())


#******************************************************************************************************
#---------------------------------
# DIAGRAMA DE CAJA TOTAL GRAFICA |
#---------------------------------

# fig4,ax4=plt.subplots(1,1)
# ax4.hist(data['diagnosis'],align='mid')

nrow=2
ncol=5
# fig1, axes1=plt.subplots(nrow,ncol)
# for i in range(0,nrow):
#     for j in range(0,ncol):
#         if (1+(i+1)*(i+j))<=len(data.columns):          
#             data.boxplot(ax=axes1[i,j],column=data.columns[1+j+i*ncol])
# fig1.suptitle('Maligno y benigno')

# fig12, axes12=plt.subplots(nrow,ncol)
# for i in range(0,nrow):
#     for j in range(0,ncol):
#         if (1+(i+1)*(i+j))<=len(data.columns):         
#             data.boxplot(ax=axes12[i,j],column=data.columns[1+j+i*ncol])
# fig12.suptitle('Maligno y benigno')

# fig13, axes13=plt.subplots(nrow,ncol)
# for i in range(0,nrow):
#     for j in range(0,ncol):
#         if (1+(i+1)*(i+j))<=len(data.columns):         
#             data.boxplot(ax=axes13[i,j],column=data.columns[1+j+i*ncol])
# fig13.suptitle('Maligno y benigno')

#-----------------------------------
# DIAGRAMA DE CAJA MALIGNO y BENIGNO GRAFICA |
#-----------------------------------

data_neg=data.iloc[np.where(data['diagnosis']=='B')[0]]
data_pos=data.iloc[np.where(data['diagnosis']=='M')[0]]
fig2, axes2=plt.subplots(nrow,ncol)
# fig2.suptitle('Maligno')
for i in range(0,nrow):
    for j in range(0,ncol):
        if (1+(i+1)*(i+j))<=len(data_pos.columns):      
            data_pos.boxplot(ax=axes2[i,j],column=data_pos.columns[1+j+i*ncol],color='red',positions=[0])
            data_neg.boxplot(ax=axes2[i,j],column=data_neg.columns[1+j+i*ncol],color='green',positions=[1])
            axes2[i,j].set_title(data_pos.columns[1+j+i*ncol])
            axes2[i,j].set(xticklabels=[])
        


fig21, axes21=plt.subplots(nrow,ncol)
# fig21.suptitle('Maligno')
for i in range(0,nrow):
    for j in range(0,ncol):
        if (1+(i+1)*(i+j))<=len(data_pos.columns):        
            data_pos.boxplot(ax=axes21[i,j],column=data_pos.columns[1+j+(i+2)*ncol],color='red',positions=[0])
            data_neg.boxplot(ax=axes21[i,j],column=data_neg.columns[1+j+(i+2)*ncol],color='green',positions=[1])
            axes21[i,j].set_title(data_pos.columns[1+j+(i+2)*ncol])
            axes21[i,j].set(xticklabels=[])

fig22, axes22=plt.subplots(nrow,ncol)
# fig22.suptitle('Maligno')
for i in range(0,nrow):
    for j in range(0,ncol):
        if (1+(i+1)*(i+j))<=len(data_pos.columns):      
            data_pos.boxplot(ax=axes22[i,j],column=data_pos.columns[1+j+(i+4)*ncol],color='red',positions=[0])
            data_neg.boxplot(ax=axes22[i,j],column=data_neg.columns[1+j+(i+4)*ncol],color='green',positions=[1])
            axes22[i,j].set_title(data_pos.columns[1+j+(i+4)*ncol])
            axes22[i,j].set(xticklabels=[])

#-----------------------------------
# DIAGRAMA DE CAJA BENIGNO GRAFICA |
#-----------------------------------


# fig3, axes3=plt.subplots(nrow,ncol)
for i in range(0,nrow):
    for j in range(0,ncol):
        if (1+(i+1)*(i+j))<=len(data_neg.columns):        
            data_neg.boxplot(ax=axes2[i,j],column=data_neg.columns[1+j+i*ncol],color='green')

# fig31, axes31=plt.subplots(nrow,ncol)
for i in range(0,nrow):
    for j in range(0,ncol):
        if (1+(i+1)*(i+j))<=len(data_neg.columns):        
            data_neg.boxplot(ax=axes21[i,j],column=data_neg.columns[1+j+i*ncol],color='green')

# fig32, axes32=plt.subplots(nrow,ncol)
for i in range(0,nrow):
    for j in range(0,ncol):
        if (1+(i+1)*(i+j))<=len(data_neg.columns):        
            data_neg.boxplot(ax=axes22[i,j],column=data_neg.columns[1+j+i*ncol],color='green')

plt.show()



print('FIN')
