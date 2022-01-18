from crypt import methods
from email.mime import image
from flask import Flask,jsonify,request
from main import getPrediction

app = Flask(__name__)
@app.route("/predict-digit",methods = ["POST"])
def predict_data():
    img = request.files.get("digit")
    prediction = getPrediction(img)
    return jsonify({
        "prediction":prediction
    }),200
if __name__ == "__main__":
    app.run(debug = True)