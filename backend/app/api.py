from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.web_classifier import router as web_classification_router
from app.routers.users_controller import router as users_router
from app.models.model_up import ModelLoader


app = FastAPI()


# @app.on_event("startup")
# def load_model():
#     app.state.model = ModelLoader(
#         path="models/tf/iris", name="iris", backend="tensorflow"
#     )
    # app.state.model = ModelLoader(
    #     path="models/sklearn/iris_model.sav", name="iris", backend="sklearn"
    # )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


app.include_router(
    web_classification_router,
    prefix="/web_clasification",
    tags=["web_clasification"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    users_router,
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)


@app.get("/")
async def root():
    return {"message": "Welcom to app web-classification"}
