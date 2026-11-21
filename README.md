# Accident Risk Detector Platform

Classifies accident risk (low / medium / high) from vehicle sensor readings:
speed (km/h), shock (m/s²), rotation rate (deg/s) and a panic button.

## Files
- `app.py` - dashboard, model training and prediction (Streamlit + scikit-learn)
- `data/Dataset.csv` - 2000 sensor readings with risk labels
- `requirements.txt` - libraries

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Model
Random Forest (3 classes, class-balanced), 80/20 stratified split.
Compared against a simple hand-written rule.

## Limitations
- The dataset appears simulated: a 3-line rule reaches about 96% accuracy,
  so model accuracy does not prove real-world performance.
- 'low' is only about 5% of rows (imbalanced).
- The panic button is pressed in about 49% of rows, which is unrealistic.
- The live sensor feed is simulated. Real ESP32/Arduino input is future work.
- Random split, not time-based.

## Live dashboards 

