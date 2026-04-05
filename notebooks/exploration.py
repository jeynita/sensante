import pandas as pd
import matplotlib.pyplot as plt

# 1. Charger les données
df = pd.read_csv("data/patients_dakar.csv")

print("=" * 50)
print(" SENSANTE - Exploration du dataset ")
print("=" * 50)

# 2. Dimensions du dataset
print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

# 3. Aperçu des 5 premieres lignes
print("\n--- Extrait des 5 premiers patients ---")
print(df.head())

# 4. Statistiques de santé
print("\n--- Statistiques descriptives ---")
print(df.describe().round(2))

## 5. Répartition par diagnostic
print("\n--- Répartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts() 

for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f" {diag:12s} : {count:3d} patients ({pct:.1f}%)")

# 6. Repartition par region
print("\n--- Repartition par region (top 5) ---")
region_counts = df["region"].value_counts().head(5)
for region, count in region_counts.items():
    print(f" {region:15s} : {count:3d} patients ") 

# 7. TEMPERATURE MOYENNE PAR diagnostic
print("\n--- Temperature moyenne par diagnostic ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items():
    print(f" {diag:12s} : {temp:.1f} C") 
    
print ( f"\n{ '= ' * 50}")
print (" Exploration terminee !")
print (" Prochain lab : entrainer un modele ML")
print ( f"{ '= ' * 50}")


# Créer une figure avec 3 graphiques côte à côte
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
import pandas as pd
import matplotlib.pyplot as plt

# 1. Charger les données
df = pd.read_csv("data/patients_dakar.csv")

print("=" * 50)
print(" SENSANTE - Exploration du dataset ")
print("=" * 50)

# 2. Dimensions du dataset
print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

# 3. Aperçu des 5 premieres lignes
print("\n--- Extrait des 5 premiers patients ---")
print(df.head())

# 4. Statistiques de santé
print("\n--- Statistiques descriptives ---")
print(df.describe().round(2))

## 5. Répartition par diagnostic
print("\n--- Répartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts() 

for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f" {diag:12s} : {count:3d} patients ({pct:.1f}%)")

# 6. Repartition par region
print("\n--- Repartition par region (top 5) ---")
region_counts = df["region"].value_counts().head(5)
for region, count in region_counts.items():
    print(f" {region:15s} : {count:3d} patients ") 
    
# EXERCICE 1 : Analyse par sexe et diagnostic 
print("\n Repartition par sexe et diagnostic")
sexe_diag = df.groupby(["sexe", "diagnostic"]).size()
print(sexe_diag)
print(df.groupby(["sexe", "diagnostic"]).size())

# 7. TEMPERATURE MOYENNE PAR diagnostic
print("\n--- Temperature moyenne par diagnostic ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items():
    print(f" {diag:12s} : {temp:.1f} C") 
    
print ( f"\n{ '= ' * 50}")
print (" Exploration terminee !")
print (" Prochain lab : entrainer un modele ML")
print ( f"{ '= ' * 50}")


# Créer une figure avec 3 graphiques côte à côte
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# --- Graphique 1 : Diagnostics (Barres) ---
diag_counts = df['diagnostic'].value_counts()
colors = ['#28a745', '#dc3545', '#007bff', '#fd7e14'] # Vert, Rouge, Bleu, Orange
diag_counts.plot(kind='bar', ax=ax1, color=colors)
ax1.set_title("Répartition des Diagnostics")
ax1.set_ylabel("Nombre de Patients")

# --- Graphique 2 : Température par Diagnostic (Histogramme) ---
for diag in df['diagnostic'].unique():
    subset = df[df['diagnostic'] == diag]
    ax2.hist(subset['temperature'], bins=15, alpha=0.5, label=diag)
ax2.set_title("Distribution des Températures")
ax2.set_xlabel("Température (°C)")
ax2.legend()

# --- Graphique 3 : Top 5 Régions (Barres horizontales) ---
region_counts = df['region'].value_counts().head(5)
region_counts.sort_values().plot(kind='barh', ax=ax3, color='#008b8b')
ax3.set_title("Top 5 Régions")
ax3.set_xlabel("Nombre de Patients")

plt.tight_layout()
plt.show() # Cette commande ouvre la fenêtre avec les graphiques
