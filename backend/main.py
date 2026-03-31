from fastapi import FastAPI
import re
from huggingface_hub import hf_hub_download
from pydantic import BaseModel,Field,computed_field,field_validator
from typing import Annotated
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware



REPO_ID = "ArmanKhan01/Emotion_Detection_BiLSTM"

# download files
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="emotion_bilstm_model.h5"
)

tokenizer_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="tokenizer.pkl"
)

label_encoder_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="label_encoder.pkl"
)

# load
model = load_model(model_path)

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

with open(label_encoder_path, "rb") as f:
    le = pickle.load(f)

max_len=178



class TextInput(BaseModel):
    text:Annotated[str,Field(...,min_length=10,max_length=500)]
 
    #to avoid number and all in input
    @field_validator('text')
    def text_must_contain_letters(cls, v):
        if not re.search(r'[a-zA-Z]', v):
            raise ValueError('Text must contain actual words')
        return v




app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/predict')
def hello(user_input:TextInput):
# Now i have got Inpur i will preprocess it 
    text=user_input.text
    text=text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    seq = tokenizer.texts_to_sequences([text])
    pad_sequence=pad_sequences(seq, maxlen=max_len, padding='pre')

    #now for predict
    pred=model.predict(pad_sequence)

    label = str(le.inverse_transform([np.argmax(pred)])[0])
    confidence = round(float(np.max(pred))*100,2)

    return {
        "emotion": label,
        "confidence":confidence
} 

@app.get('/health')
def health():
    return {"status": "ok"}
