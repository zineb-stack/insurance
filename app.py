import joblib
import pandas as pd
import streamlit as st

# Modèle et scaler sauvegardés à la fin du notebook insurance.ipynb
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title='Prédiction des frais médicaux', page_icon='💊')
st.title('💊 Prédiction des frais médicaux')
st.write("Estimation des frais médicaux annuels d'un assuré à partir de son profil.")

# Encodage identique au LabelEncoder du notebook (ordre alphabétique)
SEXE = {'Femme': 0, 'Homme': 1}
FUMEUR = {'Non': 0, 'Oui': 1}
REGION = {'Nord-Est': 0, 'Nord-Ouest': 1, 'Sud-Est': 2, 'Sud-Ouest': 3}

col1, col2 = st.columns(2)
with col1:
    age = st.slider('Âge', 18, 64, 40)
    bmi = st.number_input('IMC (BMI)', 15.0, 55.0, 28.5, step=0.1)
    children = st.selectbox("Nombre d'enfants", [0, 1, 2, 3, 4, 5])
with col2:
    sex = st.radio('Sexe', list(SEXE), horizontal=True)
    smoker = st.radio('Fumeur', list(FUMEUR), horizontal=True)
    region = st.selectbox('Région (États-Unis)', list(REGION))


def predire(fumeur):
    data = pd.DataFrame([{
        'age': age,
        'sex': SEXE[sex],
        'bmi': bmi,
        'children': children,
        'smoker': fumeur,
        'region': REGION[region],
    }])
    data_scaled = scaler.transform(data)
    return float(model.predict(data_scaled)[0])


if st.button('Prédire'):
    charges = predire(FUMEUR[smoker])
    st.subheader(f'Frais médicaux estimés : {charges:,.0f} $ par an'.replace(',', ' '))

    # Comparaison avec le même profil, fumeur / non-fumeur
    autre = 1 - FUMEUR[smoker]
    autre_charges = predire(autre)
    texte = 'non-fumeur' if autre == 0 else 'fumeur'
    st.write(f'Le même profil {texte} : environ {autre_charges:,.0f} $ par an.'.replace(',', ' '))

st.caption('Projet académique. Données : Medical Cost Personal Dataset (Kaggle). Estimation indicative.')
