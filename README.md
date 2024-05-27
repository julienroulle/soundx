# Documentation Technique du Projet "soundx"

## Aperçu du Projet

Le projet "soundx" est une application de traitement audio utilisant diverses bibliothèques Python pour l'analyse, la visualisation et l'apprentissage automatique.

## Structure des Fichiers

### Fichiers Principaux

- **pyproject.toml**: Fichier de configuration pour Poetry, spécifiant les dépendances et les métadonnées du projet.
- **README.md**: Fichier de documentation de base.
- **Makefile**: Définit des commandes pour synchroniser les données, entraîner et évaluer les modèles.

## Dépendances

Le projet utilise les bibliothèques suivantes :

- **pandas**
- **streamlit**
- **boto3**
- **matplotlib**
- **librosa**
- **python-dotenv**
- **tensorflow-hub**
- **seaborn**
- **datasets**
- **evaluate**
- **ipykernel**
- **ipywidgets**
- **tensorflow**
- **tensorflow-macos** (en développement)

## Commandes Makefile

- **sync**: Synchronise les données depuis un bucket S3.
- **train**: Lance l'entraînement du modèle.
- **eval**: Évalue le modèle.

## Architecture des Données

L'architecture des données du projet "soundx" est conçue pour gérer efficacement le traitement, l'analyse et l'apprentissage automatique sur des ensembles de données audio. Voici un aperçu détaillé de l'architecture des données :

### Flux de Données

1. **Acquisition des Données**

   - Les données audio sont synchronisées depuis un bucket S3 `soundx-audio-dataset` vers le répertoire local [data/raw] à l'aide de la commande `sync` du `Makefile`.

2. **Prétraitement des Données**

   - Les données brutes sont prétraitées pour les rendre compatibles avec les modèles d'apprentissage automatique. Cela peut inclure des étapes telles que la normalisation, le découpage en segments, et l'extraction de caractéristiques audio (features).

3. **Stockage des Données Prétraitées**

   - Les données prétraitées sont stockées dans un répertoire dédié, `data/processed`, pour une utilisation ultérieure dans les étapes d'entraînement et d'évaluation des modèles.

4. **Entraînement des Modèles**

   - Les données prétraitées sont utilisées pour entraîner les modèles d'apprentissage automatique. Le script `src/multi_train.py` est exécuté via la commande `train` du `Makefile`.

5. **Évaluation des Modèles**

   - Les modèles entraînés sont évalués à l'aide des données de test pour mesurer leur performance. Le script `src/multi_eval.py` est exécuté via la commande `eval` du `Makefile`.

### Répertoires de Données

- **data/raw**: Contient les données audio brutes synchronisées depuis le bucket S3.
- **data/processed**: Contient les données prétraitées prêtes pour l'entraînement et l'évaluation des modèles.

### Scripts de Traitement

- **src/multi_train.py**: Script pour l'entraînement des modèles d'apprentissage automatique.
- **src/multi_eval.py**: Script pour l'évaluation des modèles.

### Bibliothèques Utilisées

- **librosa**: Pour le traitement audio et l'extraction de caractéristiques.
- **pandas**: Pour la manipulation et l'analyse des données.
- **tensorflow**: Pour la construction et l'entraînement des modèles d'apprentissage automatique.
- **boto3**: Pour l'interaction avec AWS S3.

## Interface Streamlit

Le projet "soundx" utilise Streamlit pour créer des interfaces utilisateur interactives pour le traitement et l'analyse des données audio. Voici un aperçu détaillé des interfaces Streamlit utilisées dans le projet :

### Structure et fonctionnalités de l'Interface

L'interfaces Streamlit est une application multi-pages avec les sections suivantes : - **Accueil**: Page d'accueil de l'application. - **Dataset**: Page récapitulative du dataset utilisé. - **Evaluation**: Page d'évaluation des modèles sur chaque classe. - **Model**: Page de prédiction des modèles. - **Train**: Page d'entraînement des modèles.

## Modèle Utilisé et Prétraitement dans [multi_train.py]

### Prétraitement des Données

Le script [multi_train.py] gère le prétraitement des données audio avant l'entraînement du modèle. Voici les étapes principales :

1. **Chargement et Resampling des Fichiers Audio**:

   - Les fichiers audio sont chargés et resamplés à 16 kHz.

2. **Extraction des Embeddings**:

   - Utilisation du modèle YAMNet pour extraire les embeddings des fichiers audio.

3. **Création des Datasets TensorFlow**:
   - Les données sont transformées en datasets TensorFlow, avec des étapes de cache, de shuffle, et de batch.

### Modèle Utilisé

Le modèle utilisé est un réseau de neurones séquentiel construit avec TensorFlow. Il se compose de couches denses et utilise une couche personnalisée pour la réduction de la moyenne.

1. **Définition du Modèle**:

   - Le modèle est défini avec une couche d'entrée, deux couches denses, et une couche de sortie.

2. **Compilation et Entraînement**:

   - Le modèle est compilé avec une fonction de perte `SparseCategoricalCrossentropy` et l'optimiseur `Adam`. L'entraînement utilise un callback pour l'arrêt anticipé.

3. **Conversion en TFLite**:
   - Le modèle entraîné est converti en format TensorFlow Lite pour une utilisation optimisée.

### Sauvegarde et Téléchargement des Modèles

Les modèles et les fichiers associés sont sauvegardés localement et téléchargés sur un bucket S3.

Cette structure permet un flux de travail efficace pour le prétraitement, l'entraînement, et le déploiement des modèles d'apprentissage automatique dans le projet "soundx".

## Auteur

- **Julien Roullé** (ju.roulle@gmail.com)
