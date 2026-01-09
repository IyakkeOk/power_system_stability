from xgboost import XGBClassifier


def build_xgboost_model():
    """
    XGBoost classifier for power system stability prediction.
    """

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    return model
