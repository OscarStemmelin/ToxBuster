from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

def model_choice(model):
    SEED = 42
    match model:
        case "Logistic":
            return LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)
        case "Multinomial":
            return MultinomialNB()
        case "RandomForest":
            return RandomForestClassifier(max_depth=7, random_state=SEED)
        case "GradientBoosting":
            return GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=SEED)
        case "XGBoost":
            return XGBClassifier()
        case _:
            raise ValueError("Please select a model from the list")