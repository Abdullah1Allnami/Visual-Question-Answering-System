import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import random
import numpy as np
from transformers import CLIPImageProcessor, GPT2Tokenizer

# Color configurations
COLORS = {
    "red": (230, 25, 75),
    "green": (60, 180, 75),
    "blue": (0, 130, 200),
    "yellow": (255, 225, 25),
    "magenta": (240, 50, 230),
    "cyan": (70, 240, 240)
}

SHAPES = ["circle", "square", "triangle"]

QUADRANTS = {
    "top-left": (55, 55),
    "top-right": (165, 55),
    "bottom-left": (55, 165),
    "bottom-right": (165, 165)
}

def draw_shape(draw, shape, center, size, color):
    """Draws a geometric shape on a PIL ImageDraw object."""
    x, y = center
    r = size // 2
    if shape == "circle":
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    elif shape == "square":
        draw.rectangle([x - r, y - r, x + r, y + r], fill=color)
    elif shape == "triangle":
        # Equilateral triangle vertices
        p1 = (x, y - r)
        p2 = (x - int(r * 0.866), y + r // 2)
        p3 = (x + int(r * 0.866), y + r // 2)
        draw.polygon([p1, p2, p3], fill=color)

def generate_vqa_sample():
    """
    Generates a single synthetic image containing random colored shapes placed
    in quadrants, along with multiple visual question-answer pairs about it.
    Returns:
        image: PIL.Image
        qa_pairs: list of dicts with {"question": str, "answer": str}
    """
    # Create black canvas
    img = Image.new("RGB", (224, 224), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    
    # Randomly populate quadrants
    quadrant_occupants = {}
    active_quads = random.sample(list(QUADRANTS.keys()), k=random.randint(2, 4))
    
    for quad in active_quads:
        shape = random.choice(SHAPES)
        color_name = random.choice(list(COLORS.keys()))
        color_rgb = COLORS[color_name]
        
        draw_shape(draw, shape, QUADRANTS[quad], size=40, color=color_rgb)
        quadrant_occupants[quad] = {"shape": shape, "color": color_name}
        
    qa_pairs = []
    
    # Question type 1: "What shape is in the [quadrant]?"
    for quad, info in quadrant_occupants.items():
        qa_pairs.append({
            "question": f"what shape is in the {quad}",
            "answer": info["shape"]
        })
        
    # Question type 2: "What color is the [shape]?" (if shape is unique)
    shapes_present = [info["shape"] for info in quadrant_occupants.values()]
    for quad, info in quadrant_occupants.items():
        if shapes_present.count(info["shape"]) == 1:
            qa_pairs.append({
                "question": f"what color is the {info['shape']}",
                "answer": info["color"]
            })
            
    # Question type 3: "Is there a [color] [shape]?"
    # Positive case
    for quad, info in quadrant_occupants.items():
        qa_pairs.append({
            "question": f"is there a {info['color']} {info['shape']}",
            "answer": "yes"
        })
    # Negative case (choose random shape/color combination not present)
    all_combinations = [(c, s) for c in COLORS.keys() for s in SHAPES]
    present_combinations = [(info["color"], info["shape"]) for info in quadrant_occupants.values()]
    absent_combinations = [comb for comb in all_combinations if comb not in present_combinations]
    if absent_combinations:
        c, s = random.choice(absent_combinations)
        qa_pairs.append({
            "question": f"is there a {c} {s}",
            "answer": "no"
        })
        
    # Question type 4: Counting shapes
    qa_pairs.append({
        "question": "how many shapes are there",
        "answer": {1: "one", 2: "two", 3: "three", 4: "four"}[len(quadrant_occupants)]
    })
    
    # Choose one random QA pair to return for this sample training instance
    # (or we can return all, but returning one pair per image is common for basic datasets)
    selected_qa = random.choice(qa_pairs)
    return img, selected_qa

class VQASyntheticDataset(Dataset):
    """
    PyTorch Dataset loading generated synthetic VQA samples, tokenizing prompts and targets.
    """
    def __init__(self, num_samples=500, vit_model_name="openai/clip-vit-base-patch32", llm_model_name="gpt2", max_length=40):
        self.num_samples = num_samples
        self.max_length = max_length
        
        # Load processors/tokenizers
        self.image_processor = CLIPImageProcessor.from_pretrained(vit_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(llm_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Pre-generate samples so training uses fixed static batches
        self.samples = []
        for _ in range(num_samples):
            img, qa = generate_vqa_sample()
            self.samples.append((img, qa))
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        img, qa = self.samples[idx]
        question = qa["question"]
        answer = qa["answer"]
        
        # 1. Preprocess image
        # image_features is dict containing 'pixel_values': tensor (3, 224, 224)
        image_inputs = self.image_processor(images=img, return_tensors="pt")
        pixel_values = image_inputs.pixel_values.squeeze(0) # shape (3, 224, 224)
        
        # 2. Tokenize prompt & answer
        prompt_text = f"Question: {question} Answer: "
        full_text = f"Question: {question} Answer: {answer}{self.tokenizer.eos_token}"
        
        prompt_enc = self.tokenizer(prompt_text, add_special_tokens=False)
        full_enc = self.tokenizer(full_text, add_special_tokens=False)
        
        prompt_len = len(prompt_enc["input_ids"])
        full_len = len(full_enc["input_ids"])
        
        # Labels should mask out the prompt using -100, leaving only answer tokens
        labels = [-100] * prompt_len + full_enc["input_ids"][prompt_len:]
        
        # Pad or truncate to max_length
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        else:
            pad_len = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len
            
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "question": question,
            "answer": answer
        }
