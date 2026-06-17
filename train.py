import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import VQASyntheticDataset
from model import ViTLLMFusionModel
from transformers import GPT2Tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT + LLM (Fusion Head) VQA Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for projection head")
    parser.add_argument("--train_size", type=int, default=400, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--vit_model", type=str, default="openai/clip-vit-base-patch32", help="Hugging Face ViT checkpoint")
    parser.add_argument("--llm_model", type=str, default="gpt2", help="Hugging Face GPT-2/decoder checkpoint")
    parser.add_argument("--use_mlp", action="store_true", default=True, help="Use MLP instead of linear projection head")
    parser.add_argument("--unfreeze_llm", action="store_true", help="Unfreeze LLM layers during training")
    parser.add_argument("--save_path", type=str, default="checkpoints/vqa_fusion_model.pt", help="Path to save best model checkpoint")
    return parser.parse_args()

import re

def normalize_answer(s):
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', '', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Clean whitespace
    s = " ".join(s.split())
    return s

def evaluate(model, dataloader, tokenizer, device):
    """
    Evaluates the model on validation set using normalized exact-match accuracy of generated answers.
    """
    model.eval()
    correct = 0
    total = 0
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            questions = batch["question"]
            answers = batch["answer"]
            
            # For generation, we need to tokenize the questions separately
            # Format as: "Question: {question} Answer: "
            prompts = [f"Question: {q} Answer: " for q in questions]
            
            # Tokenize prompts
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Generate predictions
            # Batch generation: loop for simplicity since generated answers are short
            for i in range(pixel_values.size(0)):
                pred_tokens = model.generate_answer(
                    pixel_values=pixel_values[i:i+1],
                    question_input_ids=input_ids[i:i+1],
                    question_attention_mask=attention_mask[i:i+1],
                    max_new_tokens=10,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode prediction
                pred_text = normalize_answer(tokenizer.decode(pred_tokens[0], skip_special_tokens=True))
                gt_text = normalize_answer(answers[i])
                
                # Check match (exact matching)
                if pred_text == gt_text:
                    correct += 1
                total += 1
                
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

def main():
    args = parse_args()
    
    # Setup hardware acceleration device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    # Initialize tokenizer (required for evaluation)
    tokenizer = GPT2Tokenizer.from_pretrained(args.llm_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Datasets
    print(f"Generating synthetic datasets (Train: {args.train_size}, Val: {args.val_size})...")
    train_dataset = VQASyntheticDataset(
        num_samples=args.train_size,
        vit_model_name=args.vit_model,
        llm_model_name=args.llm_model
    )
    val_dataset = VQASyntheticDataset(
        num_samples=args.val_size,
        vit_model_name=args.vit_model,
        llm_model_name=args.llm_model
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize Model
    print("Loading pre-trained models and setting up Fusion Head...")
    model = ViTLLMFusionModel(
        vit_model_name=args.vit_model,
        llm_model_name=args.llm_model,
        use_mlp=args.use_mlp
    )
    
    if args.unfreeze_llm:
        print("Unfreezing LLM parameters for fine-tuning...")
        model.unfreeze_llm()
        # Separate learning rates: smaller for LLM, larger for projection head
        params = [
            {"params": model.projector.parameters(), "lr": args.lr},
            {"params": model.llm.parameters(), "lr": args.lr * 0.1}
        ]
    else:
        print("Keeping LLM frozen. Training only the Projection Head...")
        params = model.projector.parameters()
        
    model.to(device)
    
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    
    best_val_acc = -1.0
    
    print("\nStarting VQA Fusion training loop:")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                text_labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Average Training Loss: {avg_loss:.4f}")
        
        # Evaluate validation performance
        val_acc = evaluate(model, val_loader, tokenizer, device)
        print(f"Validation Exact-Match Accuracy: {val_acc:.2f}%")
        
        # Save checkpoint if accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best accuracy! Saving model weight checkpoint to {args.save_path}")
            # Ensure the output directory exists
            if os.path.dirname(args.save_path):
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            # We save the state dict. Note that if only projector is trained, saving just
            # projector state dict is much lighter (~2.3MB). But saving the full model state dict
            # ensures we can load it back seamlessly.
            torch.save(model.state_dict(), args.save_path)
            
    print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {os.path.abspath(args.save_path)}")

if __name__ == "__main__":
    main()
