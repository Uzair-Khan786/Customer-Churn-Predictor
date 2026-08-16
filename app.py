from flask import Flask, render_template, request,jsonify
import joblib
import pandas as pd
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model('model.json')

preprocessor = joblib.load('preprocessor.joblib')

app = Flask(__name__)

FEATURE_NAMES = [
    'SeniorCitizen','tenure', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'TechSupport', 'StreamingTV', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges'
]

def recommend_action(customer, churn_probability):
    actions = []

    if churn_probability < 40:
        return ["No immediate action required. Continue regular engagement."]

    if customer["Contract"] == "Month-to-month":
        actions.append("Offer a discounted long-term contract.")

    if customer["MonthlyCharges"] > 70:
        actions.append("Offer a personalized pricing or loyalty discount.")

    if customer["TechSupport"] == "No":
        actions.append("Offer technical support or a support upgrade.")

    if customer["OnlineSecurity"] == "No":
        actions.append("Offer a security service bundle.")

    if customer["tenure"] <= 6:
        actions.append("Provide a new-customer retention offer.")


    if not actions:
        actions.append("Send a personalized retention offer and monitor engagement.")

    return actions[:2]


@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict',methods=['POST'])
def predict():
    try:
        customer_dict = {
            "SeniorCitizen" : request.form.get('SeniorCitizen', 'No'),
            "tenure": float(request.form.get('tenure', 0) or 0),
            "MultipleLines": request.form.get('MultipleLines', 'No'),
            "InternetService": request.form.get('InternetService', 'No'),
            "OnlineSecurity": request.form.get('OnlineSecurity', 'No'),
            "OnlineBackup": request.form.get('OnlineBackup', 'No'),
            "TechSupport": request.form.get('TechSupport', 'No'),
            "StreamingTV": request.form.get('StreamingTV', 'No'),
            "Contract": request.form.get('Contract', 'Month-to-month'),
            "PaperlessBilling": request.form.get('PaperlessBilling', 'No'),
            "PaymentMethod": request.form.get('PaymentMethod', 'Electronic check'),
            "MonthlyCharges": float(request.form.get('MonthlyCharges', 0) or 0),
        }
    
        df = pd.DataFrame([customer_dict], columns=FEATURE_NAMES)

        df['SeniorCitizen'] = df['SeniorCitizen'].replace({'Yes' : 1,'No' : 0})

        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[-1,6,12,24,73],
            labels=['0-6 months','6-12 months','1-2 years','2+ years']
        ).astype(str)

        df['high_risk'] = ((df['tenure'] <= 6) & (df['TechSupport'] == 'No')).astype(int)

        df['Senior_Pressure'] = df['MonthlyCharges'] * df['SeniorCitizen']
        df.drop('SeniorCitizen',axis=1,inplace=True)

        processed_data = preprocessor.transform(df)
        feature_names = preprocessor.get_feature_names_out()
        processed_df = pd.DataFrame(processed_data, columns=feature_names)

        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0]
        churn_probability = round(probability[1] * 100,2)

        output = 'High Churn Risk' if prediction == 1 else 'Low Churn Risk'
        recommended_actions = recommend_action(customer_dict, churn_probability)
    
        result = {
            "status": "success",
            "prediction": output,
            "confidence": churn_probability,
            "recommended_actions": recommended_actions
        }

    except Exception as e:
        result = {
            "status": "error",
            "message": "Error 400 : Invalid Input"
        }

    return render_template('index.html', result=result)



if __name__ == "__main__":
    app.run()
