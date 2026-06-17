import os
import io
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import GPT2Tokenizer, CLIPImageProcessor
from model import ViTLLMFusionModel

app = Flask(__name__)

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS) for inference.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA) for inference.")
else:
    device = torch.device("cpu")
    print("Using CPU for inference.")

# Global model variables
model = None
tokenizer = None
image_processor = None
CHECKPOINT_PATH = "vqa_fusion_model.pt"
if os.path.exists(os.path.join("checkpoints", "vqa_fusion_model.pt")):
    CHECKPOINT_PATH = os.path.join("checkpoints", "vqa_fusion_model.pt")


def load_vqa_resources():
    global model, tokenizer, image_processor
    print("Loading resources... This may take a moment.")
    
    # Load tokenizer and processor
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load model architecture
    model = ViTLLMFusionModel(
        vit_model_name="openai/clip-vit-base-patch32",
        llm_model_name="gpt2"
    )
    
    # Load fine-tuned weights if available
    if os.path.exists(CHECKPOINT_PATH):
        try:
            print(f"Loading fine-tuned weights from {CHECKPOINT_PATH}...")
            state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
            model.load_state_dict(state_dict)
            print("Fine-tuned weights loaded successfully!")
        except Exception as e:
            print(f"Error loading checkpoint weights: {e}. Model will use default initialization.")
    else:
        print("No fine-tuned checkpoint found. Model will use default/random projection weights.")
        
    model.to(device)
    model.eval()

# Load resources before the first request
load_vqa_resources()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "question" not in request.form:
        return jsonify({"error": "Missing image or question"}), 400
        
    image_file = request.files["image"]
    question = request.form["question"].strip()
    
    if not image_file or not question:
        return jsonify({"error": "Invalid image or empty question"}), 400
        
    try:
        # Load and convert image to RGB
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess image
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
        
        # Tokenize question
        prompt_text = f"Question: {question} Answer: "
        prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        question_input_ids = prompt_enc["input_ids"].to(device)
        question_attention_mask = prompt_enc["attention_mask"].to(device)
        
        # Generate prediction
        generated_ids = model.generate_answer(
            pixel_values=pixel_values,
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            max_new_tokens=15,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode prediction
        answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006, debug=True)
