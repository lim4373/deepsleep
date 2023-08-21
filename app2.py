import numpy as np
from flask import Flask, request, render_template
import pickle
#from model_final import rf_model, gb_model, meta_learner

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("xgb_hyperopt.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/index")
def index():
    return render_template("index.html")

@flask_app.route("/landing")
def landing():
    return render_template("landing.html")

@flask_app.route("/generic")
def generic():
    return render_template("generic.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    if prediction[0] < 10:
        prediction_text = "당신의 예상 AHI 지수는 {:.2f}로 건강한 수면 상태입니다.".format(prediction[0])
    elif prediction[0] < 20:
        prediction_text = "당신의 예상 AHI 지수는 {:.2f}로 경도의 수면무호흡증 상태입니다.".format(prediction[0])
    else:
        prediction_text = "당신의 예상 AHI 지수는 {:.2f}로 중증의 수면무호흡증 상태입니다.".format(prediction[0])
    
    return render_template("landing.html", prediction_text=prediction_text)

if __name__ == "__main__":
    flask_app.run(debug=True)

#템플릿 출처 : https://html5up.net