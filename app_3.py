import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

st.set_page_config(
    page_title="Prédictions Multi-Modèles",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Application de Prédictions Multi-Modèles")
st.markdown("---")

@st.cache_resource
def load_models():
    """Charge les modèles pré-entraînés"""
    try:
        models = {
            'Random Forest': joblib.load('models/random_forest.pkl'),
            'K Nearest Neighbors': joblib.load('models/knn.pkl'),
            'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
            'Decision Tree': joblib.load('models/decision_tree.pkl'),
            'Support Vector Machine': joblib.load('models/suport_vector_machine.pkl'),
        }
        return models
    except FileNotFoundError as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        st.stop()

@st.cache_resource
def load_scaler():
    """Charge le scaler pré-entraîné si disponible"""
    try:
        scaler = joblib.load('models/scaler.pkl')
        return scaler
    except FileNotFoundError:
        scaler = StandardScaler()
        scaler.mean_ = np.array([1.0, 0.3, 0.4, 0.5])
        scaler.scale_ = np.array([1.0, 0.46, 0.49, 0.5])
        return scaler

with st.spinner("Chargement des modèles..."):
    models = load_models()
    scaler = load_scaler()

st.success("Modèles chargés avec succès!")

st.subheader("Faire des prédictions")

model_choice = st.selectbox(
    "Choisir un modèle pour les prédictions:",
    list(models.keys()),
    help="Sélectionnez le modèle que vous souhaitez utiliser"
)

st.write(f"**Modèle sélectionné:** {model_choice}")

feature_columns = ['Oldpeak', 'ChestPainType_ASY', 'ExerciseAngina_Y', 'ST_Slope_Flat']
feature_descriptions = {
    'Oldpeak': 'Dépression ST induite par l\'exercice (0-4)',
    'ChestPainType_ASY': 'Type de douleur thoracique asymptomatique (0=Non, 1=Oui)',
    'ExerciseAngina_Y': 'Angine induite par l\'exercice (0=Non, 1=Oui)',
    'ST_Slope_Flat': 'Pente du segment ST plate (0=Non, 1=Oui)'
}

st.write("### Saisie des caractéristiques")
prediction_inputs = {}
cols = st.columns(2)

for i, feature in enumerate(feature_columns):
    with cols[i % 2]:
        if feature in ['ChestPainType_ASY', 'ExerciseAngina_Y', 'ST_Slope_Flat']:
            prediction_inputs[feature] = st.selectbox(
                f"{feature}:",
                [0, 1],
                index=0,
                help=feature_descriptions[feature]
            )
        else:
            prediction_inputs[feature] = st.number_input(
                f"{feature}:",
                min_value=0.0,
                max_value=6.0,
                value=1.0,
                step=0.1,
                help=feature_descriptions[feature]
            )

if st.button("Faire la prédiction", type="primary"):
    input_data = np.array([list(prediction_inputs.values())])
    
    selected_model = models[model_choice]
    
    if model_choice in ['Logistic Regression', 'Support Vector Machine']:
        input_data_scaled = scaler.transform(input_data)
        prediction = selected_model.predict(input_data_scaled)[0]
        
        if hasattr(selected_model, 'predict_proba'):
            probability = selected_model.predict_proba(input_data_scaled)[0]
            prob_positive = probability[1]
            prob_negative = probability[0]
        else:
            decision_score = selected_model.decision_function(input_data_scaled)[0]
            prob_positive = 1 / (1 + np.exp(-decision_score))
            prob_negative = 1 - prob_positive
    else:
        prediction = selected_model.predict(input_data)[0]
        
        if hasattr(selected_model, 'predict_proba'):
            probability = selected_model.predict_proba(input_data)[0]
            prob_positive = probability[1]
            prob_negative = probability[0]
        else:
            prob_positive = float(prediction)
            prob_negative = 1 - prob_positive
    
    st.markdown("---")
    st.subheader("Résultats de la prédiction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Prédiction",
            value=f"Classe {int(prediction)}",
            delta="Positif" if prediction == 1 else "Négatif"
        )
    
    with col2:
        st.metric(
            label="Probabilité Classe Positive",
            value=f"{prob_positive:.1%}",
            delta=f"{prob_positive:.4f}"
        )
    
    with col3:
        st.metric(
            label="Probabilité Classe Négative",
            value=f"{prob_negative:.1%}",
            delta=f"{prob_negative:.4f}"
        )
    

    st.subheader("Détails de la prédiction")
    
    with st.expander("Valeurs d'entrée utilisées"):
        input_df = pd.DataFrame({
            'Caractéristique': list(prediction_inputs.keys()),
            'Valeur': list(prediction_inputs.values()),
            'Description': [feature_descriptions[k] for k in prediction_inputs.keys()]
        })
        st.dataframe(input_df, use_container_width=True)
    
    st.subheader("Interprétation")
    
    if prob_positive > 0.7:
        st.success("**Risque élevé** - Le modèle prédit une forte probabilité de classe positive")
    elif prob_positive > 0.3:
        st.warning("**Risque modéré** - Le modèle indique une probabilité intermédiaire")
    else:
        st.info("**Risque faible** - Le modèle prédit une faible probabilité de classe positive")
    
    confidence = max(prob_positive, prob_negative)
    st.write(f"**Niveau de confiance:** {confidence:.1%}")
    
    if confidence > 0.8:
        st.success("Prédiction très fiable")
    elif confidence > 0.6:
        st.warning("Prédiction moyennement fiable")
    else:
        st.error("Prédiction peu fiable")

st.markdown("---")
st.subheader("Informations sur les modèles")

model_info = {
    'Random Forest': 'Ensemble de arbres de décision qui vote pour la prédiction finale',
    'K Nearest Neighbors': 'Classifie basé sur les k voisins les plus proches',
    'Logistic Regression': 'Régression logistique pour la classification binaire',
    'Decision Tree': 'Arbre de décision simple basé sur des règles',
    'Support Vector Machine': 'Machine à vecteurs de support pour la classification'
}

for model, description in model_info.items():
    st.write(f"**{model}:** {description}")

with st.expander("Informations techniques"):
    st.write("""
    **Caractéristiques utilisées:**
    - **Oldpeak**: Dépression du segment ST induite par l'exercice par rapport au repos
    - **ChestPainType_ASY**: Indicateur de douleur thoracique asymptomatique
    - **ExerciseAngina_Y**: Présence d'angine induite par l'exercice
    - **ST_Slope_Flat**: Pente du segment ST plate pendant l'exercice
    
    **Notes:**
    - Les modèles Logistic Regression et SVM utilisent des données normalisées
    - Les probabilités sont calculées différemment selon le type de modèle
    - La confiance est basée sur la probabilité maximale entre les deux classes
    """)

st.markdown("---")
st.markdown("**Application créée avec Streamlit | Modèles basés sur scikit-learn**")