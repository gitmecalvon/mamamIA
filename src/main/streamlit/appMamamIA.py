import streamlit as st

import joblib  as jl


col=['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'area_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst']
modnames=['mlp_final.pkl','svm_final.pkl','LR_final.pkl']



   



def cargaModelos ():
    print('Preparando el guardado de Modelos ' )
    mlp =    jl.load(modnames[0])
    svm =    jl.load(modnames[1])
    lr =    jl.load(modnames[2])
    print('Modelo Guardado')
    print('*************')

def contruyeFormulario():
    st.title("Mama mIA")
    st.write("Algoritmo de ayuda a la predicción de deteccion de Cáncer de mama")
    form = st.form(key="formulario")
    radius_mean = form.text_input(label="radius_mean")
    texture_mean = form.text_input(label="texture_mean")
    perimeter_mean = form.text_input(label="perimeter_mean")
    area_mean = form.text_input(label="area_mean")
    compactness_mean = form.text_input(label="compactness_mean")
    concavity_mean = form.text_input(label="concavity_mean")

    concave_points_mean = form.text_input(label="concave_points_mean")
    area_se = form.text_input(label="area_se")
    radius_worst = form.text_input(label="radius_worst")
    texture_worst= form.text_input(label="texture_worst")

    perimeter_worst = form.text_input(label="perimeter_worst")
    area_worst = form.text_input(label="area_worst")
    smoothness_worst = form.text_input(label="Variable 3")
    compactness_worst = form.text_input(label="Variable 4")
    concavity_worst = form.text_input(label="Variable 5")
    concavepoints_worst = form.text_input(label="Variable 6")
    # valor = form.slider ("deslizable",-5,+5,0.0001)


    submit = form.form_submit_button(label="Predicción")

    if submit:
            # st.write("slider", valor)
            st.write ([[radius_mean, texture_mean, perimeter_mean ,
            area_mean ,  compactness_mean ,  concavity_mean ,
            concave_points_mean ,  area_se ,  radius_worst ,  texture_worst ,
            perimeter_worst ,  area_worst ,  smoothness_worst ,
            compactness_worst ,  concavity_worst ,  concavepoints_worst ]])
            st.write ("cargando modelo")

            algoritmo=jl.load(modnames[2])
            resultado=999
            st.write (resultado)
            resultado = algoritmo.predict ([[radius_mean, texture_mean, perimeter_mean ,area_mean ,  compactness_mean ,  concavity_mean ,
            concave_points_mean ,  area_se ,  radius_worst ,  texture_worst ,perimeter_worst ,  area_worst ,  smoothness_worst ,
            compactness_worst ,  concavity_worst ,  concavepoints_worst ]] )
            st.write("aqui que vamos")
            st.write (resultado)
            st.write (jl)
            st.write ("Los datos aportados hacen indicar ")
            if resultado ==1:
                st.write ("Maligno")
            else:
                st.write ("Benigno")

def main():
    cargaModelos()
    contruyeFormulario()
 
if __name__ == '__main__':
    main()







        








