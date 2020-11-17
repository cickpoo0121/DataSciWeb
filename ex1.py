
import streamlit as st
import pandas as pd
import pickle


st.write("""
# My First Web Application
Let's enjoy **data science** project!
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')
# st.sidebar.radio('Sex', ['Male','Female','Infant'])


# st.header('Application of Abalone\'s Age Prediction:')
# st.subheader('User Input:')


def get_unput():
    v_sex = st.sidebar.radio('Sex', ['Male', 'Female', 'Infant'])
    v_length = st.sidebar.slider('Length', 0.075, 0.745, 0.506790)
    v_diameter = st.sidebar.slider('Length', 0.055000, 0.600000, 0.400600)
    v_height = st.sidebar.slider('Length', 0.010000, 0.240000, 0.138800)
    v_whole_weight = st.sidebar.slider('Length', 0.002000, 2.550000, 0.785165)
    v_shucked_weight = st.sidebar.slider(
        'Length', 0.001000, 1.070500, 0.308956)
    v_viscera_weight = st.sidebar.slider(
        'Length', 0.000500, 0.541000, 0.170249)
    v_shell_weight = st.sidebar.slider('Length', 0.001500, 1.005000, 0.249127)

    if v_sex == 'Male':
        v_sex = 'M'
    elif v_sex == 'Female':
        v_sex = 'F'
    else:
        v_sex = 'I'

    data = {'Sex': v_sex, 'Length': v_length, 'Diameter': v_diameter, 'Height': v_height, 'Whole_weight': v_whole_weight,
            'Shucked_weight': v_shucked_weight, 'Viscera_weight': v_viscera_weight, 'Shell_weight': v_shell_weight}
    data_df = pd.DataFrame(data, index=[0])
    return data_df


df = get_unput()
st.write(df)

data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample], axis=0)

cat_data = pd.get_dummies(df[['Sex']])

X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1]
X_new = X_new.drop(columns=['Sex'])
# st.write(cat_data)
# st.write(X_new)

load_nor = pickle.load(open('normalization.pkl', 'rb'))
X_new = load_nor.transform(X_new)
# st.write(X_new)


load_knn = pickle.load(open('best_knn.pkl', 'rb'))
prediction = load_knn.predict(X_new)

st.write(prediction)




