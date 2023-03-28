from funtions.extraccion_text import ScrapTool, clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

import mlflow
import pandas as pd

model = joblib.load('C:/Trabajo_Final_ProyectosIA/backend/mlflow/mlruns/965330960676835535/20bfa703e63d40dc978d192fe1930480/artifacts/SVM/model.pkl')

model1 = pd.read_csv('X_train.csv', header = 0)
X_train = model1['Website Cleaned Text']

tf_idf_vectorizer = TfidfVectorizer()
fited_vectorizer = tf_idf_vectorizer.fit(X_train)

scraptool = ScrapTool()

# mlflow.set_tracking_uri("http://localhost:4000/")
model_name = "Web-classifier-best-model"
stage = "Production"

URL_EX ='https://www.espn.com.co/futbol/resultados'
web = dict(ScrapTool.visit_url(scraptool,website_url=URL_EX))
text = (clean_text(web['website_text']))
X_test = fited_vectorizer.transform([text])
X_test = pd.DataFrame(X_test.toarray(), columns=fited_vectorizer.get_feature_names_out())

print(model.predict(X_test))