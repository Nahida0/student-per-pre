from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        study_hours = float(data['study_hours'])
        attendance  = float(data['attendance'])
        prev_grade  = float(data['prev_grade'])

        if not (0 <= attendance <= 100):
            return jsonify({"error": "Attendance must be 0-100"}), 400
        if not (0 <= prev_grade <= 100):
            return jsonify({"error": "Grade must be 0-100"}), 400

        features = np.array([[study_hours, attendance, prev_grade]])
        result = model.predict(features)[0]
        prediction = "Pass" if result == 1 else "Fail"

        return jsonify({"prediction": prediction}), 200

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)