import torch
import torch.nn as nn
from transformers import CLIPVisionModel, GPT2LMHeadModel

class VQAProjector(nn.Module):
    """
    Projects ViT visual embeddings into the LLM token embedding space.
    Can be configured as a linear projection or a multi-layer perceptron (MLP).
    """
    def __init__(self, vit_dim=768, llm_dim=768, use_mlp=True):
        super().__init__()
        if use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(vit_dim, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim)
            )
        else:
            self.proj = nn.Linear(vit_dim, llm_dim)
            
    def forward(self, x):
        return self.proj(x)

class ViTLLMFusionModel(nn.Module):
    """
    Visual Question Answering model using a pre-trained Vision Transformer (ViT)
    and a pre-trained Large Language Model (GPT-2) connected via a projection head.
    """
    def __init__(self, vit_model_name="openai/clip-vit-base-patch32", llm_model_name="gpt2", use_mlp=True):
        super().__init__()
        # Load pre-trained models
        self.vit = CLIPVisionModel.from_pretrained(vit_model_name)
        self.llm = GPT2LMHeadModel.from_pretrained(llm_model_name)
        
        # Freeze ViT and LLM weights to train only the projection head (efficient fusion head training)
        # We can also choose to unfreeze LLM weights for fine-tuning.
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # Projection Head
        vit_dim = self.vit.config.hidden_size
        llm_dim = self.llm.config.n_embd
        self.projector = VQAProjector(vit_dim=vit_dim, llm_dim=llm_dim, use_mlp=use_mlp)
        
    def forward(self, pixel_values, input_ids, attention_mask, text_labels=None):
        """
        Forward pass of the fusion model.
        Args:
            pixel_values: Tensor of shape (batch_size, 3, 224, 224) - visual input
            input_ids: Tensor of shape (batch_size, seq_len) - prompt + answer token ids
            attention_mask: Tensor of shape (batch_size, seq_len) - text attention mask
            text_labels: Tensor of shape (batch_size, seq_len) - text labels (ignore elements marked -100)
        """
        device = pixel_values.device
        batch_size = pixel_values.size(0)
        
        # 1. Get visual embeddings from ViT
        # output shape: (batch_size, num_patches, vit_dim)
        vit_outputs = self.vit(pixel_values)
        image_features = vit_outputs.last_hidden_state
        num_patches = image_features.size(1)
        
        # 2. Project visual features to LLM embedding dimension
        # output shape: (batch_size, num_patches, llm_dim)
        visual_embeddings = self.projector(image_features)
        
        # 3. Get text embeddings from LLM
        # output shape: (batch_size, seq_len, llm_dim)
        text_embeddings = self.llm.transformer.wte(input_ids)
        
        # 4. Concatenate visual and text embeddings along sequence dimension
        # output shape: (batch_size, num_patches + seq_len, llm_dim)
        concat_embeds = torch.cat([visual_embeddings, text_embeddings], dim=1)
        
        # 5. Concatenate attention masks
        # Visual tokens are always active (mask = 1)
        visual_mask = torch.ones(batch_size, num_patches, dtype=torch.long, device=device)
        concat_mask = torch.cat([visual_mask, attention_mask], dim=1)
        
        # 6. Setup labels for language modeling loss
        if text_labels is not None:
            # Visual tokens are ignored in loss computation (-100)
            visual_labels = torch.full((batch_size, num_patches), -100, dtype=torch.long, device=device)
            concat_labels = torch.cat([visual_labels, text_labels], dim=1)
        else:
            concat_labels = None
            
        # 7. Setup position IDs explicitly
        concat_position_ids = torch.arange(0, num_patches + input_ids.size(1), dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 8. Pass concatenated embeddings through LLM decoder
        outputs = self.llm(
            inputs_embeds=concat_embeds,
            attention_mask=concat_mask,
            position_ids=concat_position_ids,
            labels=concat_labels,
            return_dict=True
        )
        
        return outputs

    @torch.no_grad()
    def generate_answer(self, pixel_values, question_input_ids, question_attention_mask, max_new_tokens=20, eos_token_id=50256):
        """
        Autoregressively generate an answer token-by-token from an image and question.
        """
        self.eval()
        device = pixel_values.device
        batch_size = pixel_values.size(0)
        
        # 1. Extract visual embeddings
        vit_outputs = self.vit(pixel_values)
        image_features = vit_outputs.last_hidden_state
        num_patches = image_features.size(1)
        visual_embeddings = self.projector(image_features)
        
        # 2. Extract text embeddings
        text_embeddings = self.llm.transformer.wte(question_input_ids)
        
        # 3. Initialize sequence
        current_embeds = torch.cat([visual_embeddings, text_embeddings], dim=1)
        
        # Initialize mask
        visual_mask = torch.ones(batch_size, num_patches, dtype=torch.long, device=device)
        current_mask = torch.cat([visual_mask, question_attention_mask], dim=1)
        
        # Initialize position IDs
        current_position_ids = torch.arange(0, num_patches + question_input_ids.size(1), dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        generated_tokens = []
        
        for i in range(max_new_tokens):
            # Forward pass
            outputs = self.llm(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                position_ids=current_position_ids,
                return_dict=True
            )
            
            # Predict next token (argmax over vocabulary)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # shape: (batch_size, 1)
            
            generated_tokens.append(next_token)
            
            # Stop if EOS token is generated
            if next_token.item() == eos_token_id:
                break
                
            # Embed the generated token and append to sequence
            next_embed = self.llm.transformer.wte(next_token)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            
            # Extend mask
            next_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            current_mask = torch.cat([current_mask, next_mask], dim=1)
            
            # Extend position IDs
            next_pos = torch.full((batch_size, 1), num_patches + question_input_ids.size(1) + i, dtype=torch.long, device=device)
            current_position_ids = torch.cat([current_position_ids, next_pos], dim=1)
            
        return torch.cat(generated_tokens, dim=-1)

    def unfreeze_llm(self):
        """
        Optional utility to unfreeze the LLM parameters for full end-to-end training.
        """
        for param in self.llm.parameters():
            param.requires_grad = True
            
    def unfreeze_vit(self):
        """
        Optional utility to unfreeze the ViT parameters for full end-to-end training.
        """
        for param in self.vit.parameters():
            param.requires_grad = True
