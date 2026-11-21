# Prédiction des frais médicaux d'assurance

Estimer les frais médicaux annuels d'un assuré à partir de son profil (âge, IMC, tabac, enfants, région) et identifier les facteurs qui font le plus varier les coûts.

**Démo en ligne :** [insurance-zineb.streamlit.app](https://insurance-zineb.streamlit.app/)

## Le problème
Les assureurs santé doivent estimer les coûts médicaux de leurs assurés pour mieux gérer les risques et proposer des tarifs adaptés.

## Les données
- Source : [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (Kaggle)
- 1 338 assurés, 7 colonnes : `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`
- 1 doublon supprimé, aucune valeur manquante

## Ce que j'ai fait
1. **Nettoyage :** doublons, valeurs manquantes, étude des valeurs aberrantes (IMC et frais) avec la méthode IQR.
2. **Analyse exploratoire :** distributions, relations entre les frais et l'âge, l'IMC, le tabac, le sexe, les enfants et la région.
3. **Préparation :** encodage des variables catégorielles (LabelEncoder) et mise à l'échelle (StandardScaler).
4. **Modélisation :** comparaison de la régression linéaire, KNN, arbre de décision, Random Forest et SVR ; le meilleur modèle est sauvegardé.
5. **Application :** interface Streamlit pour estimer les frais d'un nouveau profil.

## Résultats
| Modèle | R² | MAE ($) | RMSE ($) |
| --- | --- | --- | --- |
| **Random Forest** | 0.898 | 2 591 | 4 320 |
| Arbre de décision | 0.893 | 2 755 | 4 435 |
| KNN | 0.864 | 3 109 | 4 991 |
| Régression linéaire | 0.807 | 4 182 | 5 958 |
| SVR | -0.130 | 9 253 | 14 409 |

Le **Random Forest** est le meilleur modèle : il explique environ **90 %** de la variation des frais (R² = 0,898), avec une erreur moyenne d'environ **2 600 $**, pour des frais moyens d'environ 13 300 $.

- Le **tabac** est le facteur le plus important : à profil égal, un fumeur a des frais beaucoup plus élevés.
- Les frais augmentent avec l'**âge** et l'**IMC**.

## Outils
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit

## Structure
```
├── insurance.ipynb    # analyse complète (EDA + modèles)
├── best_model.pkl     # meilleur modèle entraîné
├── scaler.pkl         # mise à l'échelle des données
├── app.py             # application Streamlit
├── insurance.csv      # données
└── requirements.txt
```

## Lancer le projet
```bash
pip install -r requirements.txt
streamlit run app.py
```
