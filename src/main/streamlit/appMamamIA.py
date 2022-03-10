from numpy import dtype
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib  as jl


#   VALORES POR DEFECTO QUE INDICAN CELULAS NO CANCEROSAS
# radius_mean       14.12
# texture_mean      19.28
# perimeter_mean    91.96
# area_mean         551,17
#  compactness_mean  0.0092
# concavity_mean      0.061
# concave_points_mean  0.033
# area_se               24.5
# radius_worst          14.97
# texture_worst         25.41
# perimeter_worst       97.6
# area_worst            686.5
# smoothness_worst      0.1313
# compactness_worst    0.20
# concavity_worst      0.22
# concave points_worst 0.09


col=['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'area_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst']


modnames=['mlp_final.pkl','svm_final.pkl','lr_final.pkl']

#@st.cache
def getScaler():
    # Cargo el dataset para poder normalizar los valores recogidos en el formulario
    print ("cargando dataset")
    data=pd.read_csv('https://raw.githubusercontent.com/gitmecalvon/mamamIA/main/resources/data/cleaned/train_web.csv',sep=';')
    print("dataset cargado")
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


# cargandolos para poder usarlos desde un sidebar si da tiempo
def cargaModelos (indice):
    print('Preparando el guardado de Modelos ' )
    modelo=jl.load(modnames[indice])
    return modelo

def interpreta (prediccion):
    respuesta ="Los datos introducidos pronostican que son células de tipo   "
    if prediccion ==1:
        respuesta= respuesta + "Maligno"
    else:
         respuesta= respuesta + "BENIGNO"
    return respuesta


def contruyeFormulario():

    # st.set_page_config(layout="wide")

    st.title("Mama mIA")
    st.markdown('<style>body{background-color: Black;}</style>',unsafe_allow_html=True)
    html_temp = """ <div style ="background-color:Pink;padding:13px">
    <h1 style ="color:black;text-align:center;">Algoritmo de ayuda a la predicción diagnóstica del Cáncer de mama</h1>
    </div>"""    
    st.markdown(html_temp, unsafe_allow_html = True)

    st.subheader("Por favor introduzca las medidas de la muestra")
    form = st.form(key="formulario")
    # col1, col2 = form.columns(2)    # intento de dos columnas sin recurrir a html
    # with col1:
    radius_mean = form.number_input( label="Radio Promedio", min_value=0.00000, max_value=20.0,value=13.54, step=0.0001,format="%4f")
    texture_mean = form.number_input(label="Textura Promedio", min_value=0.00000, max_value=36.0,value=14.36, step=0.0001,format="%4f")
    perimeter_mean = form.number_input(label="Perímertro Promedio", min_value=0.00000, max_value=150.0,value=87.46, step=0.0001,format="%4f")
    area_mean = form.number_input(label="Área Promedio", min_value=0.00000, max_value=1600.0,value=566.3, step=0.0001,format="%4f")
    compactness_mean = form.number_input(label="Promedio de Compactabilidad", min_value=0.00000, max_value=1.0,value=0.08129, step=0.0001,format="%5f")
    concavity_mean = form.number_input(label="Promedio de Concavidad", min_value=0.00000, max_value=1.0,value=0.06664, step=0.0001,format="%5f")
    concave_points_mean = form.number_input(label="Puntos Cóncavos promedio", min_value=0.00000, max_value=1.0,value=0.04781, step=0.0001,format="%4f")
    area_se = form.number_input(label="Area Error Estandar", min_value=0.00000, max_value=150.0,value=23.56, step=0.0001,format="%4f")
    # with col2:
    radius_worst = form.number_input(label="Radio worst ", min_value=0.00000, max_value=30.0,value=15.11, step=0.0001,format="%4f")
    texture_worst= form.number_input(label="Textura worsk", min_value=0.00000, max_value=70.0,value=19.26, step=0.0001,format="%4f")
    perimeter_worst = form.number_input(label="Perimetro worst", min_value=0.00000, max_value=99.70,value=0.0092, step=0.0001,format="%4f")
    area_worst = form.number_input(label="Area ", min_value=0.00000, max_value=800.0,value=711.2, step=0.0001,format="%4f")
    smoothness_worst = form.number_input(label="Suavidad worst", min_value=0.00000, max_value=1.0,value=0.144, step=0.0001,format="%4f")
    compactness_worst = form.number_input(label="Compactabilidad worst", min_value=0.00000, max_value=2.0,value=0.1773, step=0.0001,format="%4f")
    concavity_worst = form.number_input(label="Concavidad worst", min_value=0.00000, max_value=2.0,value=0.2390, step=0.0001,format="%4f")
    concavepoints_worst = form.number_input(label="Puntos cóncavos worst", min_value=0.00000, max_value=2.0,value=0.1288, step=0.0001,format="%4f")

    submit = form.form_submit_button(label="Predicción")

    if submit:
        #  Escalamos los datos del formulario
            scaler=getScaler()
            nbnormaliz=scaler.transform  ([[radius_mean, texture_mean, perimeter_mean ,area_mean ,  compactness_mean ,  concavity_mean ,
            concave_points_mean ,  area_se ,  radius_worst ,  texture_worst ,perimeter_worst ,  area_worst ,  smoothness_worst ,
            compactness_worst ,  concavity_worst ,  concavepoints_worst ]])
          
        #    Recuperamos el modelo
            print ("cargando modelo")
            print (modnames[2])
            algoritmo=cargaModelos(2)

        #  Realizamos la prediccion
            
            print ("Preparando la prediccion...")
            prediccion=algoritmo.predict (nbnormaliz)
            print (prediccion)
            st.write ("")
            st.write (interpreta (prediccion))
            

def main():
    
    contruyeFormulario()
 
if __name__ == '__main__':
    main()

















