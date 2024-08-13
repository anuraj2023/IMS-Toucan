import io
import wave
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Utility.storage_config import MODELS_DIR
from Utility.utils import float2pcm
import uvicorn

app = FastAPI()

class TTSRequest(BaseModel):
    language: str
    text: str

# Global variable to store the TTS model
tts_model = None

def load_tts_model():
    global tts_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = ToucanTTSInterface(device=device)

@app.on_event("startup")
async def startup_event():
    load_tts_model()

@app.post("/generate_speech")
async def generate_speech(request: TTSRequest):
    global tts_model
    
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS model not initialized")
    
    tts_model.set_language(lang_id=request.language)
    
    # Generate audio without playing it
    wav, sr = tts_model(request.text,
                        view=False,
                        duration_scaling_factor=1.0,
                        pitch_variance_scale=1.0,
                        energy_variance_scale=1.0,
                        prosody_creativity=0.1)
    
    if wav is not None and sr is not None:
        # Convert numpy array to PCM
        wav_pcm = float2pcm(wav)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sr)
            wav_file.writeframes(wav_pcm.tobytes())
        
        # Reset buffer position
        wav_buffer.seek(0)
        
        # Return WAV file as streaming response
        return StreamingResponse(wav_buffer, media_type="audio/wav", headers={
            'Content-Disposition': f'attachment; filename="speech_{request.language}.wav"'
        })
    else:
        raise HTTPException(status_code=500, detail="Failed to generate audio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)