import joblib
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates', static_folder="static")


@app.route('/', methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    pkl_model = open("rf_bank_model_final.pkl", "rb")
    model = joblib.load(pkl_model)
    global result
    classes = np.array(["Charged off", "Fully Paid"])
    if request.method == "POST":
        current_loan_amount = request.form["Current Loan Amount"]
        term = request.form['Term']
        credit_score = request.form['Credit Score']
        annual_income = request.form['Annual Income']
        years_current_job = request.form['Years in current job']
        home_ownership = request.form['Home Ownership']
        purpose = request.form['Purpose']
        monthly_debt = request.form['Monthly Debt']
        years_credit_hist = request.form['Years of Credit History']
        number_of_open_accounts = request.form['Number of Open Accounts']
        number_credit_prob = request.form['Number of Credit Problems']
        current_credit_balance = request.form['Current Credit Balance']
        max_open_credit = request.form['Maximum Open Credit']
        bankruptcies = request.form["Bankruptcies"]
        tax_liens = request.form['Tax Liens']

        input_data = [{
            "Current Loan Amount": current_loan_amount,
            "Term": term,
            "Credit Score": credit_score,
            "Annual Income": annual_income,
            "Years in current job": years_current_job,
            "Home Ownership": home_ownership,
            "Purpose": purpose,
            "Monthly Debt": monthly_debt,
            "Years of Credit History": years_credit_hist,
            "Number of Open Accounts": number_of_open_accounts,
            "Number of Credit Problems": number_credit_prob,
            "Current Credit Balance": current_credit_balance,
            "Maximum Open Credit": max_open_credit,
            "Bankruptcies": bankruptcies,
            "Tax Liens": tax_liens
        }]

        data = pd.DataFrame(input_data)
        result = model.predict(data)
        proba_result = model.predict_proba(data)
        if result == 0:
            proba = ("%.2f" % proba_result[:, 0])
        else:
            proba = ("%.2f" % proba_result[:, 1])

        return render_template("result.html", script=classes[result].item(), probability=proba)


if __name__ == '__main__':
    app.run(debug=True)
