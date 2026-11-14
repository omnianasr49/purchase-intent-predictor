import pandas as pd


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Engineered session features, shared by the notebook and the app."""
    X = X.copy()
    X["TotalPages"] = X["Administrative"] + X["Informational"] + X["ProductRelated"]
    X["TotalDuration"] = X["Administrative_Duration"] + X["Informational_Duration"] + X["ProductRelated_Duration"]
    X["SecPerProductPage"] = X["ProductRelated_Duration"] / (X["ProductRelated"] + 1)
    X["HasPageValue"] = (X["PageValues"] > 0).astype(int)
    X["Weekend"] = X["Weekend"].astype(int)
    return X
