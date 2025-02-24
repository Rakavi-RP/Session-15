import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# RMSNorm: Normalization layer using RMS.
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / norm) * self.weight

# RotaryEmbedding: Implements rotary positional embeddings.
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_position = max_position        
        position = torch.arange(max_position).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', position, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(1))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(1))
    def forward(self, x, seq_len=None):
        if seq_len > self.max_position:
            seq_len = self.max_position
        return self.cos_cached[:seq_len].unsqueeze(0), self.sin_cached[:seq_len].unsqueeze(0)

# DeepSeekAttention: Implements Multi-Head Latent Attention (MHLA).
class DeepSeekAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config["num_attention_heads"]
        self.hidden_size = config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_heads
        
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_proj   = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output_proj= nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Latent projections for queries and keys
        self.latent_query = nn.Linear(self.head_dim, self.head_dim)
        self.latent_key   = nn.Linear(self.head_dim, self.head_dim)
    def forward(self, x):
        B, S, D = x.shape  # S is the sequence length (should remain unchanged)
        
        # Project and reshape to [B, S, num_heads, head_dim]
        queries = self.query_proj(x).view(B, S, self.num_heads, self.head_dim)
        keys    = self.key_proj(x).view(B, S, self.num_heads, self.head_dim)
        values  = self.value_proj(x).view(B, S, self.num_heads, self.head_dim)
        
        # Apply latent projections on queries and keys
        latent_queries = self.latent_query(queries)
        latent_keys    = self.latent_key(keys)
        
        # Compute attention scores: shape [B, num_heads, S, S]
        scores = torch.einsum('bshd,bthd->bhst', latent_queries, latent_keys) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values: result shape [B, num_heads, S, head_dim]
        attended = torch.einsum('bhst,bthd->bshd', attn, values)
        # Reshape back to [B, S, D]
        attended = attended.reshape(B, S, D)
        return self.output_proj(attended)

# FeedForward network used in experts.
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj   = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)
        self.act_fn    = nn.SiLU()
    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))

# Mixture of Experts (MoE) layer with loss-less load balancing.
class MixtureOfExperts(nn.Module):
    def __init__(self, config, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = config["hidden_size"]
        # Create experts
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(num_experts)])
        # Gating network: outputs weights for each expert
        self.gate = nn.Linear(self.hidden_size, num_experts)
        
        # Add routing bias for load balancing
        self.routing_bias = nn.Parameter(torch.zeros(num_experts))
        self.expert_load = None  # Track expert utilization

    def forward(self, x):
        # x shape: [B, S, H]
        B, S, H = x.shape
        
        # Get routing logits with bias
        routing_logits = self.gate(x) + self.routing_bias
        gates = F.softmax(routing_logits, dim=-1)  # [B, S, num_experts]
        
        # Track expert load for this batch
        self.expert_load = gates.mean(dim=(0, 1))  # Mean utilization per expert
        
        # Get outputs from each expert
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=2)  # [B, S, num_experts, H]
        
        # Combine experts with gating
        out = torch.einsum('bse,bsen->bsn', gates, expert_outs)  # [B, S, H]
        return out

    def update_bias_terms(self, expert_load):
        """Update routing biases based on expert utilization"""
        target_load = 1.0 / self.num_experts
        load_diff = expert_load - target_load
        
        # Dynamic update rate based on load imbalance
        update_rate = 0.1 * torch.abs(load_diff)
        self.routing_bias.data -= update_rate * load_diff

# Transformer block combining DeepSeekAttention and MoE.
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DeepSeekAttention(config)
        self.moe       = MixtureOfExperts(config)
        self.norm1     = RMSNorm(config["hidden_size"])
        self.norm2     = RMSNorm(config["hidden_size"])
    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x

# The main model (SmolLM2) with token embeddings, Transformer blocks, and final normalization.
class SmolLM2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config["num_hidden_layers"])])
        self.norm = RMSNorm(config["hidden_size"])
    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        # Output projection: tie with embedding weights.
        logits = F.linear(x, self.embed_tokens.weight)
        return logits

    def generate(
        self, 
        input_ids, 
        max_length, 
        min_length=None,
        num_return_sequences=1, 
        pad_token_id=None,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    ):
        self.eval()
        batch_size = input_ids.shape[0]
        min_length = min_length if min_length is not None else input_ids.shape[1]
        
        # Clear KV cache if exists
        if hasattr(self, 'kv_cache'):
            self.kv_cache = None
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                
                # Stop if all sequences have hit the pad token
                if pad_token_id is not None and (next_tokens == pad_token_id).all():
                    break
                
                # Stop if we've reached min_length
                if input_ids.shape[1] < min_length:
                    continue
                    
        return input_ids

# Utility function: creates a model from a configuration dictionary.
def create_model_from_config(config):
    model_config = {
        "vocab_size": config["model"]["model_config"]["vocab_size"],
        "hidden_size": config["model"]["model_config"]["hidden_size"],
        "num_hidden_layers": config["model"]["model_config"]["num_hidden_layers"],
        "num_attention_heads": config["model"]["model_config"]["num_attention_heads"],
        "intermediate_size": config["model"]["model_config"]["intermediate_size"],
        "num_key_value_heads": config["model"]["model_config"]["num_key_value_heads"],
        "initializer_range": config["model"]["model_config"]["initializer_range"],
        "rms_norm_eps": config["model"]["model_config"]["rms_norm_eps"]
    }
    return SmolLM2(model_config)
