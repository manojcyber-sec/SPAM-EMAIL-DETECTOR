# predict_cli.py
import joblib, os, sys
VEC_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.joblib')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'nb_model.joblib')
vec = joblib.load(VEC_PATH)
model = joblib.load(MODEL_PATH)
text = " ".join(sys.argv[1:]) or input("Paste email text: ")
X = vec.transform([text])
print("Prediction:", model.predict(X)[0])
