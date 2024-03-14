from config import paths
from data_models.data_validator import validate_data
from data_models.prediction_data_model import validate_predictions
from logger import get_logger, log_error
from prediction.predictor_model import predict_with_model

from schema.data_schema import load_json_data_schema
from utils import (
    set_seeds,
    read_csv_in_directory,
    read_json_as_dict,
    save_dataframe_as_csv,
    cast_time_col,
    ResourceTracker,
)

logger = get_logger(task_name="predict")


def run_batch_predictions(
    input_schema_dir_path: str = paths.INPUT_SCHEMA_DIR,
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    train_dir: str = paths.TRAIN_DIR,
    test_dir: str = paths.TEST_DIR,
    predictions_file_path: str = paths.PREDICTIONS_FILE_PATH,
) -> None:
    """
    Run batch predictions on test data, save the predicted probabilities to a CSV file.

    This function reads test data from the specified directory,
    loads the preprocessing pipeline and pre-trained predictor model,
    transforms the test data using the pipeline,
    makes predictions using the trained predictor model,
    adds ids into the predictions dataframe,
    and saves the predictions as a CSV file.

    Args:
        saved_schema_dir_path (str): Dir path to the saved data schema.
        model_config_file_path (str): Path to the model configuration file.
        train_dir (str): Directory path for the train data.
        test_dir (str): Directory path for the test data.
        preprocessing_dir_path (str): Path to the saved pipeline file.
        predictor_file_path (str): Path to the saved predictor model file.
        predictions_file_path (str): Path where the predictions file will be saved.
    """

    try:
        with ResourceTracker(logger, monitoring_interval=5):
            logger.info("Making batch predictions...")

            logger.info("Loading schema...")
            data_schema = load_json_data_schema(input_schema_dir_path)

            logger.info("Loading model config...")
            model_config = read_json_as_dict(model_config_file_path)

            logger.info("Setting seeds...")
            set_seeds(model_config["seed_value"])

            # we need history to make predictions
            logger.info("Loading training data...")
            train_data = read_csv_in_directory(file_dir_path=train_dir)
            logger.info("Validating training data...")
            validated_train_data = validate_data(
                data=train_data, data_schema=data_schema, is_train=True
            )

            # we need the test data to return our final predictions with right columns
            logger.info("Loading test data...")
            test_data = read_csv_in_directory(file_dir_path=test_dir)
            test_data = cast_time_col(
                test_data, data_schema.time_col, data_schema.time_col_dtype
            )
            logger.info("Validating test data...")
            validated_test_data = validate_data(
                data=test_data, data_schema=data_schema, is_train=False
            )

            logger.info("Making predictions...")
            predictions_df = predict_with_model(
                model_name=model_config["model_name"],
                num_samples=model_config["num_samples"],
                context=validated_train_data,
                forecast_length=data_schema.forecast_length,
                series_id_col=data_schema.id_col,
                time_col=data_schema.time_col,
                target_col=data_schema.target,
                prediction_field_name=model_config["prediction_field_name"],
                future_timsteps=validated_test_data[data_schema.time_col],
            )

            logger.info("Validating predictions dataframe...")
            validated_predictions = validate_predictions(
                predictions_df, data_schema, model_config["prediction_field_name"]
            )

        logger.info("Saving predictions dataframe...")
        save_dataframe_as_csv(
            dataframe=validated_predictions, file_path=predictions_file_path
        )

    except Exception as exc:
        err_msg = "Error occurred during prediction."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.PREDICT_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_batch_predictions()
