#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Déploiement du modèle KNN pour la classification cardiaque
Auteur: Assistant IA
Date: 2025
"""

import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Union
import logging
from pathlib import Path

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredicteurCardiaque:
    """
    Classe pour déployer le modèle KNN de classification cardiaque
    """
    
    def __init__(self, chemin_modele: str = "knn_fixed.pkl"):
        """
        Initialise le prédicteur avec le modèle KNN
        
        Args:
            chemin_modele (str): Chemin vers le fichier pickle du modèle
        """
        self.modele = None
        self.caracteristiques = ['Oldpeak', 'ChestPainType_ASY', 'ExerciseAngina_Y', 'ST_Slope_Flat']
        self.classes = {0: 'Pas de maladie cardiaque', 1: 'Maladie cardiaque détectée'}
        self.charger_modele(chemin_modele)
    
    def charger_modele(self, chemin_modele: str) -> None:
        """
        Charge le modèle KNN depuis le fichier pickle
        
        Args:
            chemin_modele (str): Chemin vers le fichier pickle
        """
        try:
            with open(chemin_modele, 'rb') as fichier:
                self.modele = pickle.load(fichier)
            logger.info(f"Modèle chargé avec succès depuis {chemin_modele}")
            logger.info(f"Type de modèle: {type(self.modele).__name__}")
            logger.info(f"Nombre de voisins: {self.modele.n_neighbors}")
        except FileNotFoundError:
            logger.error(f"Fichier modèle non trouvé: {chemin_modele}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise
    
    def valider_donnees(self, donnees: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Valide et formate les données d'entrée
        
        Args:
            donnees: Données d'entrée (liste, array numpy, ou DataFrame)
            
        Returns:
            np.ndarray: Données formatées pour la prédiction
        """
        if isinstance(donnees, list):
            donnees = np.array(donnees)
        elif isinstance(donnees, pd.DataFrame):
            donnees = donnees.values
        
        # Vérifier les dimensions
        if donnees.ndim == 1:
            donnees = donnees.reshape(1, -1)
        
        if donnees.shape[1] != len(self.caracteristiques):
            raise ValueError(f"Nombre de caractéristiques incorrect. Attendu: {len(self.caracteristiques)}, reçu: {donnees.shape[1]}")
        
        return donnees
    
    def predire(self, donnees: Union[List, np.ndarray, pd.DataFrame]) -> Dict:
        """
        Effectue une prédiction sur les données d'entrée
        
        Args:
            donnees: Données d'entrée
            
        Returns:
            Dict: Résultats de la prédiction avec probabilités
        """
        if self.modele is None:
            raise ValueError("Modèle non chargé. Appelez d'abord charger_modele()")
        
        try:
            # Valider et formater les données
            donnees_validees = self.valider_donnees(donnees)
            
            # Prédiction
            prediction = self.modele.predict(donnees_validees)
            probabilites = self.modele.predict_proba(donnees_validees)
            
            # Formater les résultats
            resultats = []
            for i, (pred, prob) in enumerate(zip(prediction, probabilites)):
                resultat = {
                    'prediction': int(pred),
                    'classe': self.classes[pred],
                    'probabilite_classe_0': float(prob[0]),
                    'probabilite_classe_1': float(prob[1]),
                    'confiance': float(max(prob))
                }
                resultats.append(resultat)
            
            logger.info(f"Prédiction effectuée pour {len(resultats)} échantillon(s)")
            return {'resultats': resultats}
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise
    
    def interpreter_resultat(self, resultat: Dict) -> str:
        """
        Interprète le résultat de la prédiction en français
        
        Args:
            resultat (Dict): Résultat de la prédiction
            
        Returns:
            str: Interprétation en français
        """
        classe = resultat['classe']
        confiance = resultat['confiance']
        
        if confiance >= 0.8:
            niveau_confiance = "très élevée"
        elif confiance >= 0.6:
            niveau_confiance = "élevée"
        elif confiance >= 0.4:
            niveau_confiance = "modérée"
        else:
            niveau_confiance = "faible"
        
        interpretation = f"""
        Prédiction: {classe}
        Niveau de confiance: {niveau_confiance} ({confiance:.2%})
        
        Probabilités détaillées:
        - Pas de maladie cardiaque: {resultat['probabilite_classe_0']:.2%}
        - Maladie cardiaque détectée: {resultat['probabilite_classe_1']:.2%}
        """
        
        return interpretation.strip()

def exemple_utilisation():
    """
    Exemple d'utilisation du prédicteur cardiaque
    """
    try:
        # Initialiser le prédicteur
        predicteur = PredicteurCardiaque("knn_fixed.pkl")
        
        # Exemples de données (Oldpeak, ChestPainType_ASY, ExerciseAngina_Y, ST_Slope_Flat)
        exemples_donnees = [
            [1.0, 1, 0, 0],  # Exemple 1
            [2.5, 1, 1, 1],  # Exemple 2 (plus à risque)
            [0.0, 0, 0, 0],  # Exemple 3 (moins à risque)
        ]
        
        print("=== PRÉDICTEUR CARDIAQUE KNN ===\n")
        print("Caractéristiques utilisées:")
        for i, carac in enumerate(predicteur.caracteristiques):
            print(f"  {i+1}. {carac}")
        print()
        
        # Prédictions pour chaque exemple
        for i, donnees in enumerate(exemples_donnees, 1):
            print(f"--- Exemple {i} ---")
            print(f"Données d'entrée: {donnees}")
            
            # Prédiction
            resultats = predicteur.predire([donnees])
            resultat = resultats['resultats'][0]
            
            # Interprétation
            interpretation = predicteur.interpreter_resultat(resultat)
            print(interpretation)
            print()
        
        # Prédiction pour plusieurs échantillons à la fois
        print("--- Prédiction par lots ---")
        resultats_lots = predicteur.predire(exemples_donnees)
        
        for i, resultat in enumerate(resultats_lots['resultats'], 1):
            print(f"Échantillon {i}: {predicteur.classes[resultat['prediction']]} "
                  f"(confiance: {resultat['confiance']:.2%})")
        
    except Exception as e:
        logger.error(f"Erreur dans l'exemple: {str(e)}")
        print(f"Erreur: {str(e)}")

# Interface en ligne de commande simple
def interface_console():
    """
    Interface console pour saisir des données manuellement
    """
    try:
        predicteur = PredicteurCardiaque("models\knn.pkl")
        
        print("=== INTERFACE DE PRÉDICTION CARDIAQUE ===")
        print("Veuillez saisir les valeurs pour chaque caractéristique:")
        
        while True:
            try:
                print("\nCaractéristiques à saisir:")
                donnees = []
                
                # Saisie des données
                oldpeak = float(input("1. Oldpeak (dépression ST, ex: 1.0): "))
                donnees.append(oldpeak)
                
                chest_pain = int(input("2. ChestPainType_ASY (0 ou 1): "))
                donnees.append(chest_pain)
                
                exercise_angina = int(input("3. ExerciseAngina_Y (0 ou 1): "))
                donnees.append(exercise_angina)
                
                st_slope = int(input("4. ST_Slope_Flat (0 ou 1): "))
                donnees.append(st_slope)
                
                # Prédiction
                resultats = predicteur.predire([donnees])
                resultat = resultats['resultats'][0]
                
                # Affichage des résultats
                print("\n" + "="*50)
                print("RÉSULTATS DE LA PRÉDICTION:")
                print("="*50)
                print(predicteur.interpreter_resultat(resultat))
                
                # Continuer ?
                continuer = input("\nVoulez-vous faire une autre prédiction ? (o/n): ").lower()
                if continuer != 'o':
                    break
                    
            except ValueError as e:
                print(f"Erreur de saisie: {e}. Veuillez réessayer.")
            except KeyboardInterrupt:
                print("\nArrêt du programme.")
                break
                
    except Exception as e:
        print(f"Erreur: {str(e)}")

if __name__ == "__main__":
    # Choisir le mode d'exécution
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--console":
        interface_console()
    else:
        exemple_utilisation()