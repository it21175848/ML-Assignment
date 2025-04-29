from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load SVM model
model_svm = joblib.load("model_svm.pkl")
# Load feature columns
columns = joblib.load("columns.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_form')
def predict_form():
    return render_template('index.html', columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        input_data = [float(request.form[col]) for col in columns]
        df = pd.DataFrame([input_data], columns=columns)

        # Predict with SVM model 
        prediction_svm = model_svm.predict(df)[0]

        return render_template("index.html",
                               prediction_svm=prediction_svm,
                               columns=columns)

    except Exception as e:
        return render_template("index.html", error=str(e), columns=columns)

if __name__ == "__main__":
    app.run(debug=True)
#check the commits in github