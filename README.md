# Traitement d'Images avec Florence-2

## Description

Ce projet utilise le modèle Microsoft Florence-2-large pour générer des descriptions détaillées d'images. Il est conçu pour parcourir des dossiers d'images et générer automatiquement des descriptions textuelles pour chaque image trouvée dans les dossiers se terminant par "sans_texte".

## Fonctionnalités

- Analyse automatique des dossiers d'images
- Génération de descriptions détaillées pour chaque image
- Sauvegarde des résultats en format CSV et TXT
- Journalisation complète des opérations
- Gestion de la mémoire pour les traitements volumineux
- Prise en charge de différents formats d'image (JPG, PNG, BMP, GIF)

## Prérequis

- Python 3.8+
- PyTorch
- Transformers
- Pillow (PIL)
- tqdm

## Installation

```bash
pip install torch transformers pillow tqdm
```

Pour utiliser le GPU (fortement recommandé) :
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

## Structure du projet

```
.
├── main.py              # Script principal
├── output_images/       # Dossier contenant les images à traiter
│   ├── *_sans_texte/    # Sous-dossiers à traiter (doivent se terminer par "sans_texte")
│   ├── image_descriptions_sans_texte.csv          # Fichier CSV de sortie
│   └── detailed_image_descriptions_sans_texte.txt # Fichier TXT de sortie
└── florence_processing.log  # Journal des opérations
```

## Utilisation

1. Placez vos images dans des sous-dossiers se terminant par "sans_texte" dans le dossier "output_images"
2. Exécutez le script :

```bash
python main.py
```

3. Les résultats seront sauvegardés dans :
   - `output_images/image_descriptions_sans_texte.csv`
   - `output_images/detailed_image_descriptions_sans_texte.txt`

## Détails techniques

### Classes principales

- `ProcessingResult` : Stocke les résultats du traitement d'une image
- `Florence2Error` : Exception personnalisée pour les erreurs spécifiques
- `CSVWriter` : Gère l'écriture incrémentale dans le fichier CSV

### Fonctions principales

- `setup_model()` : Initialise le modèle Florence-2 et le processeur
- `preprocess_image()` : Prépare l'image pour le traitement
- `get_image_description()` : Génère une description pour une image
- `process_image_folder()` : Traite toutes les images dans les dossiers correspondants
- `save_results_to_file()` : Sauvegarde les résultats dans un fichier texte

## Optimisations

- Utilisation du GPU lorsque disponible
- Nettoyage régulier de la mémoire pour éviter les fuites
- Prétraitement des images pour optimiser la qualité des descriptions
- Écriture incrémentale des résultats pour éviter la perte de données

## Personnalisation

Vous pouvez modifier les paramètres suivants dans le code :

- `model_id` dans `setup_model()` pour utiliser un modèle différent
- Les paramètres de génération dans `get_image_description()` pour ajuster la qualité des descriptions
- `should_process_folder()` pour changer les critères de sélection des dossiers

## Journalisation

Le script enregistre toutes les opérations dans le fichier `florence_processing.log` et affiche également les informations importantes dans la console.

## Limitations

- Nécessite une quantité importante de mémoire GPU pour les grandes images
- Le traitement peut être lent sans GPU
- Limité aux dossiers se terminant par "sans_texte"
