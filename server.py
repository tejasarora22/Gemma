import os
import io
import torch
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoProcessor, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)

print("Loading Gemma 4 E2B locally...")
model_id = "google/gemma-4-e2b-it"
processor = AutoProcessor.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
).to(device)
print(f"Model successfully locked to: {device.upper()}")

@app.route("/api/chat", methods=["POST"])
def voice_chat():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file detected"}), 400
        
    audio_file = request.files["audio"]
    
    try:
        audio_bytes = audio_file.read()
        
        # soundfile effortlessly reads the clean WAV format from RAM using io.BytesIO
        audio_data, sampling_rate = sf.read(io.BytesIO(audio_bytes))

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
        inputs = processor(text=prompt, audio=audio_data, sampling_rate=sampling_rate, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        
        input_len = inputs.input_ids.shape[-1]
        response_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        return jsonify({"response": response_text})
        
    except Exception as e:
        print(f"\n❌ Internal processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n🚀 LOCAL ENGINE OPERATIONAL")
    app.run(port=5000, debug=False)