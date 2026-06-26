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
<img width="620" height="293" alt="image" src="https://github.com/user-attachments/assets/ec70f07f-9522-4857-b5ca-e4f9d3ab9d26" />

<img width="1258" height="574" alt="image" src="https://github.com/user-attachments/assets/3cba1255-4b50-468b-83b7-f98c2cf39d5d" />

<img width="1180" height="496" alt="image" src="https://github.com/user-attachments/assets/0fbfda2c-f416-46a9-8911-ebaea17afebc" />

<img width="1279" height="474" alt="image" src="https://github.com/user-attachments/assets/a32feb0e-67c5-4868-9a13-e3460102703b" />

<img width="1186" height="474" alt="image" src="https://github.com/user-attachments/assets/6e4b8371-2a14-4ba5-9d09-5a4458408cd2" />







