import typing
import numpy as np
import mlflow
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import Depends
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from starlette.requests import Request
from pydantic import BaseModel
from app.funtions.mlflow_call import call_model_at_stage
from app.funtions.text_extraction import ScrapTool, clean_text, category
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# model = joblib.load('C:/Trabajo_Final_ProyectosIA/backend/app/routers/model.pkl')
# model1 = pd.read_csv('C:/Trabajo_Final_ProyectosIA/backend/app/routers/X_train.csv')

model = joblib.load('/code/app/routers/model.pkl')
model1 = pd.read_csv('/code/app/routers/X_train.csv')
X_train = model1['Website Cleaned Text']

scraptool = ScrapTool()
router = InferringRouter()

class Item(BaseModel):
    URL: str

@cbv(router)
class Webcontroller:

    @router.get("/")
    def welcome(self):
        return {"message": "Welcome to the Iris API"}

@router.post("/predictions")
async def create_item(item: Item):

    # mlflow.set_tracking_uri("http://localhost:4000/")
    # model_name ="Web-classifier-best-model"
    # stage = "Production"
    tf_idf_vectorizer = TfidfVectorizer()
    fitted_vectorizer = tf_idf_vectorizer.fit(X_train)


    URL_EX = dict(item).get("URL")
    web = dict(ScrapTool.visit_url(scraptool, website_url=URL_EX))
    text = (clean_text(web['website_text']))
    X_test = fitted_vectorizer.transform([text])
    X_test = pd.DataFrame(X_test.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())
    predict = model.predict(X_test)[0]
    value = category(predict)
    return value

# mlflow.set_tracking_uri("http://localhost:4000/")
# tf_idf_vectorizer = TfidfVectorizer()
# fitted_vectorizer = tf_idf_vectorizer.fit(X_train)
# model_name = "Web-classifier-best-model"
# stage = "Production"
#
# URL_EX ='https://www.espn.com/soccer/'
# web = dict(ScrapTool.visit_url(scraptool,website_url=URL_EX))
# text = (clean_text(web['website_text']))
# X_test = fitted_vectorizer.transform([text])
# X_test = pd.DataFrame(X_test.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())
#
# predict = model.predict(X_test)[0]
#
# value = category(predict)
# print(value)