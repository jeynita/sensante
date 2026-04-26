import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- PREPARATION DES DOSSIERS ---
# On s'assure que les dossiers existent AVANT de sauvegarder
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# 1. Chargement du dataset
df = pd.read_csv("data/patients_dakar.csv")

# 2. Encodage des variables catégoriques
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# 3. Définition des Features (X) et de la Cible (y)
feature_cols = [
    'age', 'sexe_encoded', 'temperature', 'tension_sys', 
    'toux', 'fatigue', 'maux_tete', 'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

# 4. Découpage du dataset (Train/Test Split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Initialisation et Entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Modèle entraîné avec succès !")

# 6. Évaluation et Sauvegarde de la Figure
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Prédictions')
plt.ylabel('Réalité')
plt.title('Matrice de Confusion : Diagnostics des Patients')

# Sauvegarde impérative de l'image
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
print("Figure sauvegardée dans figures/confusion_matrix.png")

# 7. SÉRIALISATION (C'est ici que le dossier se remplit)
# On sauvegarde le modèle ET les encodeurs pour le Lab 3
joblib.dump(model, "models/model.pkl")


# 8. Vérification finale
size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} ko")

# =================================================================
# SAUVEGARDE DES ENCODEURS ET METADATA (LAB 2)
# =================================================================

# On s'assure que le dossier models existe pour eviter toute erreur
os.makedirs("models", exist_ok=True)

# Sauvegarder les encodeurs (indispensables pour transformer les nouvelles donnees)
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")

# Sauvegarder la liste des features (pour reference lors de l'integration)
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardes avec succes dans le dossier models/.")

# =================================================================
# 7. SIMULATION DE L'API (PRÉPARATION DU LAB 3)
# =================================================================

# Charger le modele DEPUIS LE FICHIER (pas depuis la memoire)
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print("-" * 50)
print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")
print("-" * 50)

# =================================================================
# 9. RÉSULTATS DU PRÉ-DIAGNOSTIC (SIMULATION)
# =================================================================

# Un nouveau patient arrive au centre de sante de Medina
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

# 1. Encoder les valeurs categoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# 2. Preparer le vecteur de features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# 3. Predire le diagnostic et les probabilites
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

# 4. Affichage des resultats
print(f"\n--- Resultat du pre-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite : {proba_max:.1%}")

print(f"\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f" {classe:8s} : {proba:.1%} {bar}")