# Modèle de comparaison : détection de la toxicité avec l'API Moderation d'OpenAI (gratuite)
# Utilise le MÊME découpage train/test que Model.py pour que les scores soient comparables.
#
# Installation : py -m pipenv install openai
# Clé API : créer une clé sur platform.openai.com puis définir la variable d'environnement OPENAI_API_KEY
#
# L'API ne connaît pas les classes de CONDA : elle dit seulement si un message pose problème
# (harcèlement, haine, violence...). La comparaison se fait donc en BINAIRE :
# toxique (classes 0 et 1) contre non toxique (classe 2).

# ||==================||
# || F1_macro = 0.607 ||
# ||==================||

import re
import time

import pandas as pd
from nltk.tokenize import word_tokenize
from openai import OpenAI
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv("C:/Users/oscar/Desktop/ToxBuster/api_key.env")


### Paramètres
main_path = "C:/Users/oscar/Desktop/ToxBuster"
SEED = 42
MODEL = "omni-moderation-latest"
N_SAMPLE = 5000     # nombre de messages du test envoyés à l'API (None = tout le test, ~12 500 messages)
CHUNK = 32         # messages envoyés par requête (l'API accepte une liste de textes)


### Données : mêmes filtres et même découpage que Model.py
df = pd.read_csv(main_path + "/CONDA.csv", sep=";")
df = df[~df["full_text"].isnull()]
df["raw_text"] = df["full_text"]   # l'API reçoit le texte brut (ponctuation, chiffres, emojis compris)

def clean_text(text):
    return re.sub('[^a-zA-Z]', ' ', text)

df["full_text"] = df["full_text"].apply(clean_text)
df["tokens"] = df["full_text"].apply(word_tokenize)
df = df[df["tokens"].str.len() > 0]

df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)

if N_SAMPLE is not None:
    df_eval = df_test.sample(n=N_SAMPLE, random_state=SEED)
else:
    df_eval = df_test


### Appels à l'API
client = OpenAI(max_retries=5)   # réessaie automatiquement si on envoie trop de requêtes d'un coup

texts = df_eval["raw_text"].tolist()
flagged, scores = [], []
start = time.time()
for i in range(0, len(texts), CHUNK):
    response = client.moderations.create(model=MODEL, input=texts[i:i + CHUNK])
    for result in response.results:
        flagged.append(int(result.flagged))
        # Score de toxicité = la plus forte probabilité parmi toutes les catégories (harcèlement, haine...)
        scores.append(max(result.category_scores.model_dump().values()))
    print(f"{min(i + CHUNK, len(texts))}/{len(texts)} messages traités")
print(f"Terminé en {time.time() - start:.0f} s")


### Sauvegarde des prédictions (pour les comparer ensuite avec Model.py, ligne par ligne grâce à l'index)
df_eval = df_eval.assign(toxic_true=(df_eval["category_id"] != 2).astype(int),
                         pred_api=flagged, score_api=scores)
df_eval[["matchid", "line_index", "raw_text", "category_id", "toxic_true", "pred_api", "score_api"]].to_csv(
    main_path + "/predictions_api.csv", sep=";", index=True, index_label="row_id")


### Évaluation binaire : toxique (1) contre non toxique (0)
y_true = df_eval["toxic_true"].to_numpy()
y_pred = df_eval["pred_api"].to_numpy()

print(classification_report(y_true, y_pred, labels=[0, 1],
                            target_names=["non toxique", "toxique"], digits=3, zero_division=0))
print("F1 macro :", round(f1_score(y_true, y_pred, average="macro"), 3))
print("Matrice de confusion (lignes = vraie classe, colonnes = prédiction) :")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

# Le seuil d'OpenAI (flagged) est fait pour des contenus graves : il rate sans doute la toxicité légère.
# L'average precision mesure la qualité du score indépendamment du seuil choisi.
print("Average precision (aire sous la courbe précision-rappel) :",
      round(average_precision_score(y_true, df_eval["score_api"]), 3))

# Rappel par classe d'origine : quelle part de la toxicité explicite (0) et implicite (1) est détectée
print("Part des messages signalés par l'API, selon leur classe CONDA :")
print(df_eval.groupby("category_id")["pred_api"].mean().round(3))
