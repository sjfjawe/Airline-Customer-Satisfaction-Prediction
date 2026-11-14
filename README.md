# Airline Customer Satisfaction Predictor

A machine learning web application that predicts whether an airline passenger will be satisfied based on their flight experience.

**Live Demo:** [airlinesatisfactionpredictor-env.eba-mcateaih.us-east-2.elasticbeanstalk.com](http://airlinesatisfactionpredictor-env.eba-mcateaih.us-east-2.elasticbeanstalk.com)

![App Screenshot](https://github.com/sjfjawe/Airline-Customer-Satisfaction-Prediction/raw/main/screenshot.png)

## Overview

This project uses a Random Forest classifier trained on airline passenger survey data to predict customer satisfaction. The model achieves **~99% ROC-AUC** on the test set.

### Key Features
- Web UI for easy predictions
- REST API endpoint for programmatic access
- Dockerized for easy deployment
- Deployed on AWS Elastic Beanstalk

## Dataset

Source: [Hugging Face - Airline Satisfaction Analysis](https://huggingface.co/datasets/drukeroni/airline-satisfaction-analysis)

- **Training samples:** 103,904
- **Test samples:** 25,976
- **Features:** 22 (passenger info, flight details, service ratings)
- **Target:** Satisfied vs Neutral/Dissatisfied

## Model

**Algorithm:** Random Forest Classifier

**Hyperparameters:**
- `n_estimators`: 130
- `max_depth`: 10
- `min_samples_leaf`: 3

**Performance:**
| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 96.2% | 96.1% |
| ROC-AUC | 99.0% | 99.0% |

### Top Predictive Features
1. Online Boarding
2. Inflight Entertainment
3. Seat Comfort
4. Class (Business > Eco)
5. Type of Travel (Business > Personal)

## Project Structure

```
├── notebook.ipynb    # EDA, feature engineering, model training experiments
├── train.py          # Training script - loads data, trains model, saves artifacts
├── predict.py        # Inference module - loads model, runs predictions
├── serve.py          # Flask web server with UI and API
├── model.pkl         # Trained model + vectorizer + scaler
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container configuration
└── README.md
```

## Local Development

### Prerequisites
- Python 3.11+
- Docker (optional)

### Option 1: Run with Docker

```bash
# Build image
docker build -t airline-satisfaction .

# Run container
docker run -p 9696:9696 airline-satisfaction

# Open http://localhost:9696
```

### Option 2: Run with Python

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (generates model.pkl)
python train.py

# Start server
python serve.py

# Open http://localhost:9696
```

## API Usage

### Endpoint
```
POST /predict
Content-Type: application/json
```

### Request Body
```json
{
  "Gender": "Female",
  "Customer_Type": "Loyal Customer",
  "Age": 35,
  "Type_of_Travel": "Business travel",
  "Class": "Business",
  "has_delay": 0,
  "Inflight_wifi_service": 4,
  "Departure/Arrival_time_convenient": 4,
  "Ease_of_Online_booking": 4,
  "Gate_location": 3,
  "Food_and_drink": 4,
  "Online_boarding": 5,
  "Seat_comfort": 4,
  "Inflight_entertainment": 5,
  "On-board_service": 4,
  "Leg_room_service": 4,
  "Baggage_handling": 4,
  "Checkin_service": 4,
  "Inflight_service": 4,
  "Cleanliness": 4
}
```

### Response
```json
{
  "prediction": [1],
  "probability": [0.93],
  "label": ["satisfied"]
}
```

### Example curl
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"Gender":"Female","Customer_Type":"Loyal Customer","Age":35,"Type_of_Travel":"Business travel","Class":"Business","has_delay":0,"Inflight_wifi_service":4,"Departure/Arrival_time_convenient":4,"Ease_of_Online_booking":4,"Gate_location":3,"Food_and_drink":4,"Online_boarding":5,"Seat_comfort":4,"Inflight_entertainment":5,"On-board_service":4,"Leg_room_service":4,"Baggage_handling":4,"Checkin_service":4,"Inflight_service":4,"Cleanliness":4}'
```

## Deployment

### AWS Elastic Beanstalk

1. Create a ZIP of the project:
```bash
zip -r airline-app.zip Dockerfile requirements.txt train.py predict.py serve.py model.pkl
```

2. Go to AWS Elastic Beanstalk Console
3. Create new environment → Web server → Docker platform
4. Upload the ZIP file
5. Wait for deployment (~5-10 minutes)

## Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **Web:** Flask, Gunicorn
- **Container:** Docker
- **Cloud:** AWS Elastic Beanstalk

## License

MIT
