# HydroQ - Water Quality Prediction with Uncertainty (DEMO and in PROGRESS)

HydroQ is a machine learning model for predicting dissolved oxygen (DO) levels in water based on chlorophyll and temperature measurements, with uncertainty quantification.

## Table of Contents
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Command Line Options](#command-line-options)
- [Interpreting Results](#interpreting-results)
- [Development Setup](#development-setup)
- [Troubleshooting](#troubleshooting)
- [Visualizations](#visualizations)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd HydroQ
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Requirements

The model requires a CSV file with water quality measurements. By default, it looks for the following columns:

- `chl_a_mg_L`: Chlorophyll A concentration in mg/L
- `water_temp_celcius`: Water temperature in Celsius
- `doxy_mg_L`: Dissolved oxygen concentration in mg/L (target variable for training)

Example format:
```
date,location,chl_a_mg_L,water_temp_celcius,doxy_mg_L,other_columns...
2023-01-01,Lake1,5.2,23.8,7.9,...
2023-01-02,Lake1,6.1,24.2,7.6,...
```

If your data uses different column names, the training script will prompt you to specify the correct column names.

## Training the Model

To train the model with default settings:

```bash
cd wq_model
python train_wq_model.py
```

The training process will:
1. Load your data from `../data/insitu_wq_data.csv`
2. Try different encodings if there are issues reading the file
3. Ask for column names if the default names aren't found
4. Split data into training, validation, and test sets
5. Build and train a probabilistic neural network
6. Generate performance plots
7. Save the trained model and preprocessing parameters

The model and necessary files will be saved in the `model/` directory:
- `do_uncertainty_model.keras`: The trained model
- `scaler_X.pkl`: Feature scaler (for chlorophyll and temperature)
- `scaler_y.pkl`: Target scaler (for dissolved oxygen)

## Making Predictions

### Single Prediction
To predict DO levels for a single chlorophyll and temperature measurement:

```bash
python predict_do.py --values "5.2,23.8"
```

This will output the predicted DO value with uncertainty bounds.

### Batch Predictions from a CSV File
To predict DO for multiple measurements in a CSV file:

```bash
python predict_do.py --file path/to/your/data.csv --output predictions.csv
```

To visualize the predictions with uncertainty:

```bash
python predict_do.py --file path/to/your/data.csv --plot uncertainty_plot.png
```

## Command Line Options

### Training Script
The training script (`train_wq_model.py`) uses default parameters and will prompt for user input if needed.

### Prediction Script
```
usage: predict_do.py [-h] [--model-dir MODEL_DIR] (--file FILE | --values VALUES)
                    [--output OUTPUT] [--plot PLOT] [--no-header] [--chl-col CHL_COL]
                    [--temp-col TEMP_COL]

options:
  -h, --help           Show this help message
  --model-dir DIR      Directory containing the model files (default: model)
  --file FILE          CSV file with input data
  --values VALUES      Comma-separated chlorophyll and temperature (e.g., "5.2,23.8")
  --output OUTPUT      Output CSV file path for results
  --plot PLOT          Output file path for visualization
  --no-header          Do not print header row in console output
  --chl-col CHL_COL    Column name for chlorophyll values (default: chl_a_mg_L)
  --temp-col TEMP_COL  Column name for temperature values (default: water_temp_celcius)
```

## Interpreting Results

The model provides probabilistic predictions with uncertainty:

- `DO_Predicted`: The predicted dissolved oxygen value
- `DO_Uncertainty`: Standard deviation representing uncertainty
- `DO_LowerCI`: Lower bound of 95% confidence interval
- `DO_UpperCI`: Upper bound of 95% confidence interval

Higher uncertainty values indicate less confidence in the prediction.

## Development Setup

For development, we use pre-commit hooks to ensure code quality:

1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

2. Set up the git hooks:
   ```bash
   pre-commit install
   ```

3. (Optional) Run hooks against all files:
   ```bash
   pre-commit run --all-files
   ```

The pre-commit configuration will:
- Format code with black and isort
- Run flake8 for code quality checks
- Check for common issues like trailing whitespace
- Remove output from Jupyter notebooks before committing

## Troubleshooting

### Common Issues

1. **Column names not found**:
   - Use `--chl-col` and `--temp-col` to specify custom column names
   - During training, follow the prompts to specify column names

2. **Encoding issues with CSV files**:
   - The script tries multiple encodings (utf-8, latin-1, ISO-8859-1, cp1252)
   - Try converting your CSV to UTF-8 encoding if problems persist

3. **Model loading errors**:
   - Ensure TensorFlow version compatibility
   - Check that all model files are present in the model directory

4. **Missing values in data**:
   - The training script automatically removes rows with missing values
   - For prediction, ensure your input data doesn't contain NaN values

### Running Tests

To verify the model is working correctly:

```bash
# Test the model with a small sample dataset
python predict_do.py --values "5.0,20.0"

# To run a comprehensive test with visualization
python predict_do.py --file ../data/test_data.csv --plot test_results.png
```

For more assistance, please create an issue in the repository.

## Visualizations

The training process generates comprehensive visualizations in the `visualizations/` directory:

1. **Training History**:
   - `training_history.png`: Shows loss and MSE metrics over epochs
   - `learning_curve.png`: Learning curve on log scale for detailed view

2. **Prediction Performance**:
   - `prediction_scatter_plots.png`: Scatter plots of predictions vs actual values for train/val/test
   - `residual_plots.png`: Residual analysis for train/val/test sets

3. **Uncertainty Analysis**:
   - `uncertainty_vs_error.png`: How well uncertainty estimates correlate with errors
   - `uncertainty_calibration.png`: Whether uncertainty is calibrated (well-estimated)
   - `error_distribution.png`: Distribution of prediction errors

4. **Feature Analysis**:
   - `feature_correlation.png`: Correlation matrix between input features

These visualizations help assess model performance, understand prediction uncertainties, and identify potential issues.
