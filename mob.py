import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Lire le fichier CSV
energi=pd.read_csv('Financial_inclusion_dataset.csv')

# Titre principal de l'application
st.title("Analyse du dataset")

# Titre de la barre latérale
st.sidebar.title('Menu de Navigation')

# Définir les différentes pages de l'application
pages = ['Présentation', 'Nettoyage', 'Entraînement et Visualisation des données ', 'Interprétation']
page = st.sidebar.radio('Aller à la page', pages)

# Présentation
if page == pages[0]:
    st.write("# Informations sur la DataFrame")
    st.image("bank account.jpg")
    st.write("Dans ce point de contrôle, nous allons travailler sur l'ensemble de données « Inclusion financière en Afrique » qui a été fourni dans le cadre de l'Inclusion financière en Afrique hébergée par la plateforme Zindi .Description de l'ensemble de données : L'ensemble de données contient des informations démographiques et les services financiers utilisés par environ 33 600 personnes en Afrique de l'Est. Le rôle du modèle ML est de prédire quelles personnes sont les plus susceptibles d'avoir ou d'utiliser un compte bancaire.Le terme inclusion financière signifie que les particuliers et les entreprises ont accès à des produits et services financiers utiles et abordables qui répondent à leurs besoins – transactions, paiements, épargne, crédit et assurance – fournis de manière responsable et durable.")
    st.dataframe(energi.head())
    

# Nettoyage des données
elif page == pages[1]:
    st.write("## Nettoyage des données")
    
# Afficher la DataFrame initiale
    st.write("### DataFrame initiale")
    st.dataframe(energi)
    
# Encodage des variables catégorielles
    st.write("### Encodage des variables catégorielles")
    energi["bank_account"]=energi["bank_account"].map({"Yes": 1, "No": 0})
    energi["location_type"]=energi["location_type"].map({"Rural": 1, "Urban": 0})
    energi["cellphone_access"]=energi["cellphone_access"].map({"Yes": 1, "No": 0})
    
    for col in energi.select_dtypes(include=['object']).columns:   
            m = LabelEncoder()
            energi[col] = m.fit_transform(energi[col])
            energi[col] = energi[col].astype('int')
    
# Afficher la DataFrame après encodage
    st.write("### DataFrame après encodage")
    st.dataframe(energi)
    
# Supprimer des colonnes inutiles
    st.write("### Suppression des colonnes inutiles")
    mod=energi.drop(['country','uniqueid','relationship_with_head','education_level'], axis=1, inplace=True) 
    
# Remplir les valeurs manquantes avec la moyenne de la colonne
    st.write("### Remplissage des valeurs manquantes")
    energi.fillna(energi.mean(), inplace=True)
    
# Afficher la DataFrame après nettoyage
    st.write("### DataFrame après nettoyage")
    st.dataframe(energi)
       
# Détection de valeurs manquantes
    if st.checkbox('Détection de valeurs manquantes'):
        st.write("### Détection de valeurs manquantes")
        st.dataframe(energi.isnull().sum())

# Ajoutez ici d'autres sections pour la visualisation, l'entraînement, et l'interprétation
elif page == pages[2]:
    
    
    st.write("## Nettoyage des données")
    
    energi["bank_account"]=energi["bank_account"].map({"Yes": 1, "No": 0})
    energi["location_type"]=energi["location_type"].map({"Rural": 1, "Urban": 0})
    energi["cellphone_access"]=energi["cellphone_access"].map({"Yes": 1, "No": 0})
    
    for col in energi.select_dtypes(include=['object']).columns:   
            m = LabelEncoder()
            energi[col] = m.fit_transform(energi[col])
            energi[col] = energi[col].astype('int')
    
    mod=energi.drop(['country','uniqueid','relationship_with_head','education_level'], axis=1, inplace=True) 
    
    energi.fillna(energi.mean(), inplace=True)
 
    
    st.write("## Entraînement et Visualisation des données")
    
    target = 'bank_account'

    # Sélectionner les fonctionnalités
    features = energi.drop(columns=[target]).columns

    X = energi[features]
    y = energi[target]

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normaliser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Créer le modèle SVM
    model = SVR()

    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calcul des métriques de performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if st.checkbox("Afficher les métriques"):
        st.write(f'MSE: {mse}')
        st.write(f'MAE: {mae}')
        st.write(f'R2 Score: {r2}')

    # Prédiction avec des valeurs personnalisées
    st.title("Prédiction de la susceptibilité d'avoir ou d'utiliser un compte bancaire")
    st.write("Menu")
       
    st.write("location_type: ")
    
    st.write("cellphone_access")
    
    st.write("household_size")
    
    st.write("gender_of_respondent")
    
    st.write("marital_status")
    
    st.write("job_type")

    feature_values = []
    for feature in features:
        value = st.number_input(f'enter {feature}', value=0)
        feature_values.append(value)

    if st.button("Prédire"):
        # Normaliser les valeurs d'entrée
        input_data = scaler.transform([feature_values])
        prediction = model.predict(input_data)
        st.write(f'Prédiction : {prediction[0]}')

elif page == pages[3]:
    st.write("## Conclusion")
    # Ajoutez du code pour l'interprétation des résultats ici
    st.write("Selon la Banque mondiale, les trois quarts de la population pauvre du monde ne possèdent pas de compte bancaire pour des raisons ayant trait à la pauvreté, mais aussi aux frais à engager, aux distances à parcourir et à la quantité de formalités à remplir pour ouvrir un compte.Une enquête réalisée en 2011 auprès d’environ 150 000 personnes dans 148 pays révèle que 75 % des adultes gagnant moins de deux dollars par jour n’ont pas recours aux services d’institutions financières officielles. Le fait de ne pas avoir accès aux services bancaires est également lié aux inégalités de revenu : dans les pays en développement, les 20 % des adultes les plus riches ont au moins deux fois plus de chances de posséder un compte officiel que les 20 % les plus pauvres. ") 
    st.write("Les personnes qui n’ont pas accès au système bancaire officiel doivent souvent conserver  leurs gains dans des lieux vulnérable au cambriolage car n’ayant pas assez confiance aux institution bancaires .L’accès aux services financiers — ou « inclusion financière » — peut être source de transformations, car il donne aux pauvres le moyen de bâtir un avenir plus sûr. La possibilité d’épargner et d’emprunter leur permet en effet d’accumuler des actifs, de créer des entreprises, d’investir dans l’éducation, d’établir des antécédents de crédit et, finalement, d’acheter un logement.")
    
