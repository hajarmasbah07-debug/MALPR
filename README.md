# ALPR-MA — Reconnaissance Automatique de Plaques d'Immatriculation Marocaines

Ce projet implémente un système complet de lecture automatique des plaques d'immatriculation marocaines (ALPR).

## Ce que fait le projet

Le système prend en entrée une image (photo de véhicule ou de plaque) et retourne automatiquement le texte de la plaque d'immatriculation. Il fonctionne en quatre étapes enchaînées :

1. **Détection de la plaque** : un modèle YOLO26n localise et extrait la plaque dans l'image.
2. **Rectification géométrique** : la plaque extraite est redressée pour corriger les déformations de perspective.
3. **Reconnaissance des caractères (OCR)** : un modèle CRNN (CNN + BiLSTM + décodage CTC) lit les caractères de la plaque, en gérant à la fois les caractères latins et arabes présents sur les plaques marocaines.
4. **Post-traitement** : le texte brut issu du CRNN est validé et corrigé via des règles regex (conformité au format des plaques marocaines) et une distance de Levenshtein (correction des erreurs de reconnaissance par correspondance au lexique de formats valides).

## Données utilisées

Le modèle de détection a été entraîné sur un dataset d'images de plaques marocaines annotées. Le modèle OCR a été entraîné sur un dataset fusionné de 4 264 images de caractères, construit à partir de plusieurs sources pour couvrir la diversité des plaques marocaines.

## Résultats obtenus

| Module | Métrique | Résultat |
|--------|----------|----------|
| Détection (YOLO26n) | mAP@0.5:0.95 | 0.729 |
| OCR (CRNN) | Précision caractère | 98.59% |
| OCR (CRNN) | Token Error Rate | 0.0021 |

## Stack technique

Python · PyTorch · Ultralytics YOLO · OpenCV · PIL
