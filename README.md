
# Weather Predictor

A Machine Learning web application that predicts tomorrow's average temperature and the probability of rain based on current-day weather parameters.

## Features

- Predicts the average temperature for the next day.
- Estimates the probability of rain.
- Classifies whether rain is expected.
- Provides a web interface built with Flask.
- Uses pre-trained Random Forest models.


## Dataset

Weather data was obtained from [Open-Meteo](https://open-meteo.com/).

- Source: Open-Meteo Historical Weather API
- Location: 50.298767, 18.980669
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The dataset was used for training and evaluating the Machine Learning models.


## Technologies

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- joblib
- HTML
- CSS
- Chart.js

## Project Structure

```text
weather-predictor/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── predict.py
│
├── train_rain_model.py
├── train_temp_model.py
│
├── weather_model.pkl
├── rain_model.pkl
├── rain_threshold.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd weather-predictor
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the application:

```bash
python app.py
```

6. Open the application in your browser:

```text
http://127.0.0.1:5000
```

## Machine Learning Models

The project uses two Random Forest models:

### Temperature Prediction

- RandomForestRegressor
- Predicts the average temperature for the next day.
- Evaluation metric: Mean Absolute Error (MAE).

### Rain Prediction

- RandomForestClassifier
- Predicts the probability of rain on the next day.
- Uses a custom classification threshold.
- Evaluation metrics: ROC-AUC, precision, recall and F1-score.

## Model Evaluation

Current evaluation results:

| Metric | Result |
|---|---:|
| Temperature MAE | 1.65 °C |
| Rain ROC-AUC | 0.7507 |
| Rain Accuracy | 0.67 |
| Rain F1-score | 0.75 |
| Classification Threshold | 0.39 |

## Training

The training scripts use `weather.csv` as their dataset.

To train the models:

```bash
python train_temp_model.py
python train_rain_model.py
```

The scripts generate the following files:

- `weather_model.pkl`
- `rain_model.pkl`
- `rain_threshold.pkl`

The dataset is not included in this repository. Make sure to obtain it from its original source and check its license before redistribution.

## Notes

- Pre-trained model files are included, so the application can run without retraining.
- Compatible versions of scikit-learn and joblib may be required to load the saved models.
- Predictions are estimates and are intended for educational purposes.
