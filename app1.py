import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from flask import Flask, jsonify, request

# Load data
folate_data = pd.read_sas("C:\\Users\\kalen\\Downloads\\FOLATE_J.xpt")
vitamin_d_data = pd.read_sas("C:\\Users\\kalen\\Downloads\\VID_J (1).xpt")
demographics_data = pd.read_sas("C:\\Users\\kalen\\Downloads\\DEMO_J (1).xpt")
diet_day1 = pd.read_sas("C:\\Users\\kalen\\Downloads\\DR1TOT_J (1).xpt")
diet_day2 = pd.read_sas("C:\\Users\\kalen\\Downloads\\DR2TOT_J (2).xpt")

# Preprocess folate data
folate_data['LBDRFO'] = folate_data['LBDRFOSI'] / 2.265

# Preprocess vitamin D data
vitamin_d_data['Total_VitD'] = (
    vitamin_d_data['LBXVIDMS'] +  # Main Vitamin D measure
    vitamin_d_data['LBXVD2MS'] +  # Vitamin D2
    vitamin_d_data['LBXVD3MS']    # Vitamin D3
)

# Merge datasets
merged_data = folate_data.merge(vitamin_d_data, on='SEQN', how='inner')
merged_data = merged_data.merge(demographics_data, on='SEQN', how='inner')
merged_data = merged_data.merge(diet_day1, on='SEQN', how='inner')
merged_data = merged_data.merge(diet_day2, on='SEQN', how='inner')

# Handle missing values
merged_data.fillna(merged_data.median(), inplace=True)

# Feature and target selection
X = merged_data.drop(columns=['LBDRFO', 'Total_VitD', 'SEQN'])
y_folate = merged_data['LBDRFO']
y_vitamin_d = merged_data['Total_VitD']

# Train-test split
X_train_folate, X_test_folate, y_train_folate, y_test_folate = train_test_split(X, y_folate, test_size=0.2, random_state=42)
X_train_vitd, X_test_vitd, y_train_vitd, y_test_vitd = train_test_split(X, y_vitamin_d, test_size=0.2, random_state=42)

# Train CatBoost models
folate_model = CatBoostRegressor(verbose=0)
vitd_model = CatBoostRegressor(verbose=0)

folate_model.fit(X_train_folate, y_train_folate)
vitd_model.fit(X_train_vitd, y_train_vitd)

# Predictions and evaluation
folate_preds = folate_model.predict(X_test_folate)
vitd_preds = vitd_model.predict(X_test_vitd)

folate_rmse = np.sqrt(mean_squared_error(y_test_folate, folate_preds))
vitd_rmse = np.sqrt(mean_squared_error(y_test_vitd, vitd_preds))

print(f"Folate RMSE: {folate_rmse}")
print(f"Vitamin D RMSE: {vitd_rmse}")

# Flask app for alert system
app = Flask(__name__)

# Nutrient tracking and food recommendation
def load_day1_data():
    return pd.read_sas("C:\\Users\\kalen\\Downloads\\DR1IFF_J.xpt")

def recommend_foods(nutrient, threshold, day1_data):
    recommended = day1_data[day1_data[nutrient] > threshold]
    return recommended[["DR1CCMNM", "DR1CCMTX", nutrient]].sort_values(by=nutrient, ascending=False).head(10)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_features = pd.DataFrame([data])

    folate_pred = folate_model.predict(user_features)[0]
    vitd_pred = vitd_model.predict(user_features)[0]

    alerts = []
    day1_data = load_day1_data()
    recommendations = {}

    if folate_pred < 5:  # Example threshold for folate
        alerts.append("Folate levels are critically low.")
        recommendations["Folate"] = recommend_foods("DR1IFOLA", 50, day1_data).to_dict('records')

    if vitd_pred < 30:  # Example threshold for vitamin D
        alerts.append("Vitamin D levels are critically low.")
        recommendations["Vitamin D"] = recommend_foods("DR1IVD", 5, day1_data).to_dict('records')

    return jsonify({
        "Folate Prediction": folate_pred,
        "Vitamin D Prediction": vitd_pred,
        "Alerts": alerts,
        "Recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
