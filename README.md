# NYC Taxi Fare Prediction and Deployment

An end-to-end machine learning project that predicts New York City taxi fares using pickup location, drop-off location, passenger count, pickup time, and geographic distance.

The project includes data preprocessing, feature engineering, model training, model evaluation, model serialization, and deployment through a Flask web application.

## Project Overview

The application estimates the taxi fare for a New York City trip based on user-provided information.

The user enters:

* Passenger count
* Pickup date and time
* Pickup location
* Drop-off location

The application converts the location names into geographic coordinates, generates the required model features, and returns an estimated taxi fare.

## Demo

![NYC Taxi Fare Application](assets/app-screenshot.png)

> Add a screenshot of the running application to `assets/app-screenshot.png`.

**Live application:** Add deployment URL here.

## Machine Learning Workflow

```text
Raw Taxi Data
      ↓
Data Cleaning
      ↓
Geographic Filtering
      ↓
Feature Engineering
      ↓
Time-Based Train/Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Serialization
      ↓
Flask Web Application
```

## Features Used by the Model

| Feature               | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `passenger_count`     | Number of passengers                                           |
| `weekday`             | Day of the week, where Monday is 0 and Sunday is 6             |
| `hour`                | Pickup hour between 0 and 23                                   |
| `haversine_distance`  | Straight-line distance between pickup and drop-off coordinates |
| `coordinate_distance` | Sum of absolute latitude and longitude differences             |

## Feature Engineering

### Haversine Distance

The Haversine formula estimates the straight-line distance between the pickup and drop-off coordinates.

Trip distance is an important predictor because taxi fares generally increase as travel distance increases.

### Time Features

The pickup timestamp is converted into:

* Pickup hour
* Day of the week

These features help the model capture differences in travel patterns across weekdays, weekends, and different times of day.

### Geographic Filtering

The training data is filtered to remove:

* Invalid coordinates
* Trips outside the expected New York City area
* Invalid passenger counts
* Unrealistic fare values
* Records with missing information

## Model

The project uses a regression model trained with scikit-learn.

The trained model is serialized and stored as:

```text
NYC_Fare.pkl
```

The Flask application loads this model and uses it to generate fare predictions.

## Model Evaluation

A time-based validation strategy was used instead of a random train/test split.

Earlier taxi trips were used for training, while later trips were used for testing. This provides a more realistic evaluation because the model is tested against future observations.

The resulting test Root Mean Squared Error was approximately:

```text
RMSE: $5.51
```

An RMSE of $5.51 means that the model's predictions differ from actual fares by approximately $5.51, with larger prediction errors receiving additional weight.

## Training-Serving Consistency

The deployed application uses the same weekday encoding as the model training process:

```python
weekday = pickup_datetime.weekday()
```

This produces:

```text
Monday = 0
Tuesday = 1
Wednesday = 2
Thursday = 3
Friday = 4
Saturday = 5
Sunday = 6
```

Using the same feature definitions during training and prediction helps prevent training-serving skew.

## Web Application

The user interface is built with HTML and CSS.

The backend is built with Flask and performs the following steps:

1. Validates user inputs.
2. Converts location names into latitude and longitude.
3. Calculates geographic distance features.
4. Creates the model input array.
5. Generates a fare prediction.
6. Displays the estimated fare.

## Location Geocoding

The application uses the Nominatim geocoding service through the `geopy` library.

Pickup and drop-off location names are converted into geographic coordinates for every prediction.

The application handles:

* Invalid location names
* Missing inputs
* Invalid passenger counts
* Invalid date values
* Geocoding timeouts
* Temporary geocoding service failures

## Repository Structure

```text
NYC-Taxi-Fare-Deployment/
│
├── app.py
├── model.py
├── data_prep.ipynb
├── NYC_Fare.pkl
├── Procfile
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── assets/
    └── app-screenshot.png
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jags8386/NYC-Taxi-Fare-Deployment.git
cd NYC-Taxi-Fare-Deployment
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

## Example Input

```text
Passenger count: 2
Pickup date and time: 2026-07-28 10:30 AM
Pickup location: Times Square
Drop-off location: John F. Kennedy International Airport
```

The application geocodes both locations, calculates the trip features, and returns an estimated fare.

## Deployment

The repository includes a `Procfile` for deployment using Gunicorn.

```text
web: gunicorn app:app
```

The Flask development server should only be used for local testing. A production server such as Gunicorn should be used for deployment.

## Dataset

The model was developed using the NYC Taxi Fare Prediction dataset.

The complete training dataset is not stored in this repository because of its size.

To reproduce the training process:

1. Download the NYC Taxi Fare Prediction dataset.
2. Place `train.csv` in the repository root.
3. Run the preprocessing notebook or training script.
4. Save the trained model as `NYC_Fare.pkl`.

```text
train.csv
```

No confidential employer, client, customer, or production data is included in this repository.

## Environment Compatibility

The serialized model was originally created using an earlier version of scikit-learn.

Pickle models may not work correctly across significantly different scikit-learn versions. For a production modernization, the model should be retrained and validated using a currently supported Python and scikit-learn environment.

## Current Limitations

* Haversine distance measures straight-line distance rather than driving distance.
* The model does not account for traffic conditions.
* The model does not include tolls or airport surcharges.
* The model does not include real-time demand.
* The model does not use weather information.
* The application depends on an external geocoding service.
* The model was trained using historical taxi records.
* Predictions should be treated as estimates, not official fare quotes.

## Potential Improvements

Future improvements could include:

* Retraining with a modern scikit-learn version
* Comparing Linear Regression, Random Forest, XGBoost, and LightGBM
* Adding time-series-aware cross-validation
* Using actual road-network driving distance
* Adding traffic, weather, and toll information
* Adding automated unit and integration tests
* Containerizing the application with Docker
* Creating a REST API
* Adding model monitoring and drift detection
* Deploying the application to a modern cloud platform
* Adding SHAP-based model explanations

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Flask
* Geopy
* HTML
* CSS
* Gunicorn
* Jupyter Notebook
* Git
* GitHub

## Skills Demonstrated

* Data cleaning
* Exploratory data analysis
* Feature engineering
* Geographic data processing
* Regression modeling
* Time-based model validation
* Model evaluation
* Model serialization
* Flask application development
* Input validation
* Error handling
* Machine learning deployment

## Responsible Use

The output of this application is an experimental machine learning estimate.

It should not be used as an official fare calculation or as a replacement for pricing information from a transportation provider.

## Author

**Jugal Shah**

Data Scientist with experience in machine learning, statistical modeling, experimentation, forecasting, cloud data platforms, and generative AI.

GitHub: [jags8386](https://github.com/jags8386)

## License

This project is available for educational and portfolio purposes.
