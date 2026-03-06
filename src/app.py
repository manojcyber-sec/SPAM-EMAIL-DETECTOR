# app.py
from flask import Flask, request, render_template_string
import joblib, os
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'nb_model.joblib')
VEC_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.joblib')
app = Flask(__name__)
model = joblib.load(MODEL_PATH)
vec = joblib.load(VEC_PATH)
HTML = """
<!doctype html>
<title>Spam Detector</title>
<h2>Spam Email Detector</h2>
<form method="post">
  <textarea name="email" rows="6" cols="60" placeholder="Paste email text here"></textarea><br>
  <input type="submit" value="Check">
</form>
{% if result %}
  <h3>Prediction: <b>{{result}}</b></h3>
{% endif %}
"""
@app.route("/", methods=["GET","POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form.get("email","")
        X = vec.transform([text])
        pred = model.predict(X)[0]
        result = pred
    return render_template_string(HTML, result=result)
if __name__ == "__main__":
    app.run(debug=True)
