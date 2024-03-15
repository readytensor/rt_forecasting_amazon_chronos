import os
import warnings
import numpy as np
import pandas as pd
from typing import List, Union

warnings.filterwarnings("ignore")

import torch
from chronos import ChronosPipeline
from prediction.download_model import download_pretrained_model_if_not_exists

pretrained_model_path = os.path.join(os.path.dirname(__file__), "pretrained_model")

# download model if not exists
download_pretrained_model_if_not_exists(pretrained_model_path, model_name="chronos-t5-tiny")

# Check for GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("device used: ", device)

PREDICTOR_FILE_NAME = "predictor.joblib"
MODEL_PARAMS_FNAME = "model_params.save"


class Forecaster:
    """Chronos Timeseries Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    MODEL_NAME = "Chronos_Timeseries_Forecaster"

    def __init__(self, **kwargs):
        """Construct a new Chronos Forecaster."""
        self.model = ChronosPipeline.from_pretrained(
            pretrained_model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        

    def fit(self, *args, **kwargs):
        """Train the model."""
        return None

    def predict(
        self, context: np.ndarray, forecast_length: int, num_samples: int
    ) -> np.ndarray:
        """Make forecast."""
        return self.model.predict(
            context=context,
            prediction_length=forecast_length,
            num_samples=num_samples,
            temperature=0.0001,
            limit_prediction_length=False,
        )

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.MODEL_NAME}"


def preprocess_context(
    context: pd.DataFrame, series_id_col: str, target_col: str
) -> Union[List[torch.tensor], torch.tensor]:
    """Preprocess the context data."""

    grouped = context.groupby(series_id_col)

    all_ids = [i for i, _ in grouped]
    all_series = [i for _, i in grouped]

    if len(all_ids) == 1:
        processed_context = torch.tensor(
            all_series[0][target_col].to_numpy().reshape(1, -1)
        )

    else:
        processed_context = []
        for series in all_series:
            series = series[target_col].to_numpy().reshape(1, -1).flatten()
            series = torch.tensor(series)
            processed_context.append(series)

    return processed_context, all_ids


def predict_with_model(
    context: np.ndarray,
    forecast_length: int,
    series_id_col: str,
    target_col: str,
    time_col: str,
    future_timsteps: np.ndarray,
    num_samples: int,
    prediction_field_name: str,
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        TBD

    Returns:
        pd.DataFrame: The forecast.
    """

    processed_context, ids = preprocess_context(
        context, series_id_col, target_col=target_col
    )
    model = Forecaster()
    predictions = model.predict(
        context=processed_context,
        forecast_length=forecast_length,
        num_samples=num_samples,
    )
    predictions = np.array(predictions).mean(axis=1)
    predictions = predictions.flatten()

    generated_ids = [id for id in ids for _ in range(forecast_length)]
    predictions = pd.DataFrame(
        {
            series_id_col: generated_ids,
            time_col: future_timsteps,
            prediction_field_name: predictions,
        }
    )
    return predictions
