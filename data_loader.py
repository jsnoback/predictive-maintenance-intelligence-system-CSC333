"""Load, validate, and prepare predictive maintanence data.

This module contains functions for reading the AI4I 2020 dataset (ai4i2020.csv),
checking its required columns, and preparing the features and target variables for
machine learning.
"""

from pathlib import Path
import pandas as pd


# Columns used to predict machine failure.
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COLUMN = "Machine failure"

def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load the predictive maintenance dataset from a csv file.
    
    Args:
        file_path: Path to the data csv file.
        
    Returns:
        A DataFrame containing the dataset.
    """
    data = pd.read_csv(file_path)
    return data

def validate_data(data: pd.DataFrame) -> None:
    """Check that the dataset contains the required columns.
    
    Args:
        data: DataFrame to validate.
        
    Raises:
        ValueError: If any required columns are missing.
    """
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def prepare_data(
        data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare the input features and target variable.
    
    Args:
        data: DataFrame containing the machine sensor data.
        
    Returns:
        A tuple containing the numerical feature DataFrame and
        the machine failure target Series.
    """
    validate_data(data)

    # Separate input features and target variable
    features = data[FEATURE_COLUMNS].copy()
    target = data[TARGET_COLUMN].copy()

    # Convert categorical product types into numerical columns
    features = pd.get_dummies(
        features,
        columns = ["Type"],
        dtype = int,
    )

    return features, target