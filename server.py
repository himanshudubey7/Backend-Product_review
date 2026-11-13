from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from fastapi.middleware.cors import CORSMiddleware

nltk.download('stopwords')
nltk.download('punkt')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open("./CVectorizer.pkl", "rb") as f: 
    vectorizer = pickle.load(f)

with open("./TfidfTransformer.pkl", "rb") as f:
    transformer = pickle.load(f)

with open("./FakeReviewDetectionModel.pkl", 'rb') as f:
    svcmodel = pickle.load(f)

stop_words = set(stopwords.words("english"))
stemmer   = PorterStemmer()

class ReviewReq(BaseModel):
    text:str

def clean_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Lowercase
    text = text.lower()
    # Remove non-alphanumeric (punctuation)
    text = re.sub(r"[^\w\s]", "", text)
    
    # Tokenize, remove stopwords, and stem
    words = word_tokenize(text)
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(filtered_words)


@app.post("/predict")
def predict_review(req: ReviewReq):
    cleaned = clean_text(req.text)
    X_cv = vectorizer.transform([cleaned])
    x_tf = transformer.transform(X_cv)


    prediction = svcmodel.predict(x_tf)[0]

    res_label= "Original Reivew" if prediction == 1 else "Fake Review"

    return {
        "prediction": res_label,
        "processed_text": cleaned
    }

@app.get("/")
def home():
    return {"message": "hello from fastapi! Your server is running."}