"""Unit tests for the predictive maintenance data loader."""

import pandas as pd
import pytest

from data_loader import load_data, prepare_data, validate_data

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a small sample dataset for testing."""
    return pd.DataFrame({
        "Type": ["L", "M", "H"],
        "Air temperature [K]": [298.0, 299.0, 300.0],
        "Process temperature [K]": [308.0, 309.0, 310.0], 
        "Rotational speed [rpm]": [1500, 1600, 1700],
        "Torque [Nm]": [40.0, 45.0, 50.0],
        "Tool wear [min]": [10, 20, 30],
        "Machine failure": [0, 0, 1],
    })

def test_load_data(tmp_path, sample_data) -> None:
    """Test loading a csv file."""
    file_path = tmp_path / "sample.csv"
    sample_data.to_csv(file_path, index = False)

    result = load_data(file_path)

    pd.testing.assert_frame_equal(result, sample_data)

#def test_validate_data(sample_data) -> None:
