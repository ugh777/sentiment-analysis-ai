"""Load and split the sentiment dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str):
    """Load CSV dataset with 'text' and 'label' columns."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split dataframe into train/test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    return X_train, X_test, y_train, y_test
