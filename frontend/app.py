from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("high_performance_stacking.pkl", "rb"))

# Define required features
required_features = [
    "accelerations", "prolongued_decelerations",
    "abnormal_short_term_variability", "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability", "histogram_width",
    "histogram_mode", "histogram_mean", "histogram_median"
]

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    return render_template("predict.html", file_result=None, manual_result=None)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return render_template("predict.html", file_result="No file uploaded", manual_result=None)

    file = request.files["file"]
    if file.filename == "":
        return render_template("predict.html", file_result="No selected file", manual_result=None)

    # Read CSV File
    df = pd.read_csv(file)

    # Check if all required features are present
    if not all(feature in df.columns for feature in required_features):
        return render_template("predict.html", file_result="Invalid file format. Missing required columns.", manual_result=None)

    X = df[required_features]  # Extract required columns
    predictions = model.predict(X)

    # Convert numeric labels to human-readable
    label_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    predicted_classes = [label_map[pred] for pred in predictions]

    # Join predictions into a simple string
    result_text = ", ".join(predicted_classes)

    return render_template("predict.html", file_result=f"Predicted Classes: {result_text}", manual_result=None)

# Handle Manual Input Prediction
@app.route("/manual_predict", methods=["POST"])
def manual_predict():
    try:
        # Extract user input
        input_data = [float(request.form[feature]) for feature in required_features]
        input_data = np.array([input_data])  # Reshape for model input

        # Make Prediction
        prediction = model.predict(input_data)[0]

        # Convert numeric label to human-readable
        label_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}
        result = label_map[prediction]

        return render_template("predict.html", file_result=None, manual_result=result)

    except Exception as e:
        return render_template("predict.html", file_result=None, manual_result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
