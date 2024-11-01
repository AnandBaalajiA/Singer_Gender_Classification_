from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import numpy as np
import librosa
import joblib
import torch
import openunmix
import torchaudio
import simpleaudio as sa
from pydub import AudioSegment
import os

app = FastAPI()

class GenderPredictionResult(BaseModel):
    total_chunks: int
    male_count: int
    female_count: int
    male_percentage: float
    female_percentage: float
    predictions: List[str]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Audio Processing with FastAPI</title>
        </head>
        <body>
            <h1>Audio Processing with FastAPI</h1>
            
            <h2>1. Separate Audio</h2>
            <form action="/separate-audio/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="audio/*" required>
                <button type="submit">Upload and Separate</button>
            </form>
            
            <h2>2. Predict Gender</h2>
            <form action="/predict-gender/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="audio/*" required>
                <label for="chunk_duration">Chunk Duration (seconds):</label>
                <input type="number" id="chunk_duration" name="chunk_duration" value="30" required>
                <label for="overlap_duration">Overlap Duration (seconds):</label>
                <input type="number" id="overlap_duration" name="overlap_duration" value="2" required>
                <button type="submit">Upload and Predict</button>
            </form>
            
            <h2>3. Convert and Play Audio</h2>
            <form action="/convert-and-play/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="audio/*" required>
                <button type="submit">Upload and Play</button>
            </form>
        </body>
    </html>
    """

@app.post("/separate-audio/")
async def separate_audio(file: UploadFile = File(...)):
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    output_vocals = 'files/vocals.wav'
    output_accompaniment = 'files/accompaniment.wav'
    
    separator = openunmix.umxl()
    try:
        audio, sample_rate = torchaudio.load(file_location)
    except Exception as e:
        return {"error": f"Error loading audio file: {e}"}
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)
    
    with torch.no_grad():
        estimates = separator(audio)
    
    if estimates.dim() == 3:
        vocals = estimates[0, 0]
        accompaniment = estimates[0, 1]
    elif estimates.dim() == 4 and estimates.shape[1] >= 2:
        vocals = estimates[0, 0]
        accompaniment = estimates[0, 1]
    else:
        return {"error": "Estimation did not return expected shape for vocals and accompaniment."}
    
    if vocals.dim() == 3:
        vocals = vocals.squeeze(0)
    
    if accompaniment.dim() == 3:
        accompaniment = accompaniment.squeeze(0)
    
    try:
        torchaudio.save(output_vocals, vocals, sample_rate=sample_rate)
    except Exception as e:
        return {"error": f"Error saving vocal audio file: {e}"}
    
    try:
        torchaudio.save(output_accompaniment, accompaniment, sample_rate=sample_rate)
    except Exception as e:
        return {"error": f"Error saving accompaniment audio file: {e}"}
    
    return {"vocals": output_vocals, "accompaniment": output_accompaniment}

@app.post("/predict-gender/", response_model=GenderPredictionResult)
async def predict_gender(file: UploadFile = File(...), chunk_duration: int = 30, overlap_duration: int = 2):
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    y, sr = librosa.load(file_location, sr=None)
    predictions = []
    step_size = chunk_duration - overlap_duration
    
    model = joblib.load('gender_classifier.pkl')
    
    for start in range(0, len(y), step_size * sr):
        end = start + chunk_duration * sr
        chunk = y[start:end]
        
        if len(chunk) == 0:
            continue
        
        mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=128)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_mean = mfccs_mean.reshape(1, -1)
        gender_prediction = model.predict(mfccs_mean)
        prediction = "female" if gender_prediction == 1 else "male"
        predictions.append(prediction)
    
    male_count = predictions.count("male")
    female_count = predictions.count("female")
    total_predictions = male_count + female_count
    
    male_percentage = (male_count / total_predictions) * 100 if total_predictions > 0 else 0
    female_percentage = (female_count / total_predictions) * 100 if total_predictions > 0 else 0
    
    return {
        "total_chunks": total_predictions,
        "male_count": male_count,
        "female_count": female_count,
        "male_percentage": male_percentage,
        "female_percentage": female_percentage,
        "predictions": predictions
    }

@app.post("/convert-and-play/")
async def convert_and_play(file: UploadFile = File(...)):
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    try:
        audio = AudioSegment.from_wav(file_location)
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export("files/vocals_converted.wav", format="wav")
        
        wave_obj = sa.WaveObject.from_wave_file("files/vocals_converted.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        return {"message": f"Playing '{file_location}' is complete."}
    except Exception as e:
        return {"error": f"Error converting or playing audio file: {e}"}
