# Spam Email Detection System

## About the Project

This project is a machine learning based spam email detection system built using Python.
It classifies email messages as **Spam** or **Not Spam** by analyzing the content of the email text.

The system uses **TF-IDF vectorization** to convert email text into numerical features and a **Multinomial Naive Bayes classifier** to perform the classification.

The project also provides two ways to test predictions:

* Command line interface
* Simple web interface using Flask

---

## Technologies Used

* Python
* Scikit-learn
* Pandas
* Flask
* Joblib
* TF-IDF Vectorization
* Multinomial Naive Bayes

---

## Project Structure

```
SPAM-EMAIL-DETECTOR
│
├── src
│   ├── app.py
│   ├── predict_cli.py
│   └── preprocess_and_train.py
│
├── data
│   └── emails.csv
│
├── models
│   ├── nb_model.joblib
│   └── tfidf_vectorizer.joblib
│
├── README.md
└── requirements.txt
```

---

## How the System Works

1. The dataset containing labeled email messages is loaded.
2. Email text is converted into numerical features using TF-IDF.
3. A Multinomial Naive Bayes model is trained using the processed data.
4. The trained model and vectorizer are saved using Joblib.
5. New email text can be entered to predict whether it is spam or not.

---

## How to Run the Project

### Install Dependencies

```
pip install -r requirements.txt
```

### Train the Model

```
python src/preprocess_and_train.py
```

### Run the Web Application

```
python src/app.py
```

Open the browser and go to:

```
http://127.0.0.1:5000
```

Paste an email message and check the prediction.

---

## Command Line Prediction

You can also test spam detection from the terminal:

```
python src/predict_cli.py "Congratulations! You have won a free prize"
```

---

## Purpose

The goal of this project is to demonstrate how **machine learning techniques can be used to automatically detect spam emails** and improve email filtering systems.

---

## Note

This project was created for **educational purposes** to understand text classification and machine learning pipelines.
