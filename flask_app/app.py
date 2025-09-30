from flask import Flask, render_template, request
import mlflow
import dagshub
from preprocessing_utility import normalize_text
import pickle

app = Flask(__name__)

# Set up MLflow tracking URI
mlflow.set_tracking_uri('https://dagshub.com/senorhimanshu/mlops-mini-project.mlflow')
dagshub.init(repo_owner='senorhimanshu', repo_name='mlops-mini-project', mlflow=True)

# load model from model registry (globally placed here so that it loads only once)
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)
# model_version = 2

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', result = None)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]

    # clean and preprocess text
    text = normalize_text(text)

    # bow vectorization
    features = vectorizer.transform([text])

    # predict sentiment
    result = model.predict(features)

    # show result on webpage

    return render_template("index.html", result = result[0])

app.run(debug=True)