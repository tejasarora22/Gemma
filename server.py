import os
import torch
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoProcessor, AutoModelForCausalLM

# 1. Initialize Flask with Cross-Origin Resource Sharing (CORS)
app = Flask(__name__)
CORS(app)  # Connects seamlessly to your local index.html

# 2. Load Gemma 4 E2B directly onto your native GPU/CPU
print("Loading Gemma 4 E2B locally...")
model_id = "google/gemma-4-e2b-it"

processor = AutoProcessor.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)
print(f"Model successfully mapped to: {device.upper()}")

@app.route("/api/chat", methods=["POST"])
def voice_chat():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file detected"}), 400
        
    audio_file = request.files["audio"]
    temp_path = "local_input.wav"
    audio_file.save(temp_path)
    
    try:
        # Decode and resample the incoming browser format cleanly to 16kHz
        audio_data, sampling_rate = librosa.load(temp_path, sr=16000)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Listen to the user's voice and give a clean, natural response."},
                    {"type": "audio", "audio": audio_data, "sampling_rate": sampling_rate}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, audios=audio_data, sampling_rate=sampling_rate, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        
        input_len = inputs.input_ids.shape[-1]
        response_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        return jsonify({"response": response_text})
        
    except Exception as e:
        print(f"Internal processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # This keeps your directory clean even if a run errors out!
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    print("\n🚀 LOCAL ENGINE OPERATIONAL")
    print("👉 Directing Frontend API target to: http://127.0.0.1:5000/api/chat")
    app.run(port=5000, debug=False)