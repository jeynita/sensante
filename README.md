# SénSanté

> Assistant de pré-diagnostic médical pour le Sénégal, basé sur le Machine Learning.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

> Avertissement : ce projet est un exercice académique. Il ne remplace en aucun cas un avis médical.

## Présentation

SénSanté aide au pré-diagnostic de maladies courantes (paludisme, grippe, typhoïde) à partir des symptômes saisis par le patient. Un modèle de Machine Learning entraîné sur des données patients est exposé via une API FastAPI et utilisé par une interface web.

## Fonctionnalités

- Saisie des symptômes du patient depuis l'interface web
- Prédiction de la maladie la plus probable parmi : paludisme, grippe, typhoïde
- API REST documentée automatiquement (Swagger)
- <Fonctionnalité supplémentaire, ex : score de confiance de la prédiction>

## Fonctionnement

```
Interface web  -->  API FastAPI  -->  Modèle ML sérialisé  -->  Prédiction
```

## Structure du projet

```
sensante/
├── data/         # Données patients (CSV)
├── models/       # Modèle ML sérialisé
├── api/          # API FastAPI
├── frontend/     # Interface web
└── notebooks/    # Exploration des données et entraînement
```

## Technologies

- **Langage** : Python
- **Machine Learning** : <scikit-learn / autre>, modèle : <ex : Random Forest>
- **API** : FastAPI
- **Front-end** : <HTML / CSS / JavaScript>
- **Exploration** : Jupyter Notebook

## Données et modèle

- Source des données : <jeu de données fourni en cours / généré / autre>
- Nombre d'exemples : <n>
- Variables : <symptômes utilisés>
- Performance du modèle : <ex : précision de X % sur le jeu de test>

## Installation

```bash
git clone https://github.com/jeynita/sensante.git
cd sensante
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Lancer l'API :
```bash
uvicorn api.<nom_du_fichier>:app --reload
```

Documentation interactive : http://localhost:8000/docs

Ouvrir ensuite `frontend/index.html` dans le navigateur.

## Aperçu

<!-- Ajoute 2 captures dans screenshots/ : l'interface web et la page /docs de l'API -->

## Limites

- Les données utilisées ne reflètent pas un échantillon médical réel et validé : <à adapter>
- Le modèle ne couvre que trois maladies
- Aucun usage clinique

## Améliorations possibles

- Ajouter d'autres maladies courantes
- Entraîner sur des données médicales validées
- Déployer l'API et l'interface en ligne

## Contexte

Projet réalisé dans le cadre du cours « Intégration de Modèles IA » (Dr. El Hadji Bassirou TOURE), DUT Informatique 2, École Supérieure Polytechnique, UCAD.

## Auteure

Dieynaba BALDE
