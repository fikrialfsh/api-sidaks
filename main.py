from fastapi import FastAPI
import numpy
from pydantic import BaseModel
import pickle
import uvicorn

labels = {
    0: 'Hoax',
    1: 'Fakta',
    3: 'Data Tidak Ditemukan'
}

app = FastAPI()

class Response(BaseModel):
    text: str
    result: str

class Request(BaseModel):
    text: str

def process_text(text: str):
    vec = pickle.load(open('vec.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    text = text.lower()
    text_vec = vec.transform([text])
    text_tfidf = tfidf.transform(text_vec)
    return text_tfidf

model = pickle.load(open('model.pkl', 'rb'))


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict", response_model=Response)
def predict(req: Request):
    x_features = process_text(req.text)
    if numpy.all((x_features.todense() == 0)):
        label = labels[3]
    else:
        y_pred = model.predict(x_features)
        label = labels[y_pred[0]]
    response = Response(text=req.text, result=label)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
