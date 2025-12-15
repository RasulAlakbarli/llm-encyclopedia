"""
GPT-2 Model Implementation
Input -> Embedding + Pos. Encoding -> Dec Block: | LayerNorm -> MHA -> Add -> LayerNorm -> Feed Forward -> Add | -> LayerNorm -> Linear -> Softmax -> Output [Batch, Seq Len, Vocab Size]
"""

from turtle import forward
from numpy import int32
import torch
import torch.nn as nn

from positional_encoders import SinusoidalPositionalEncoding
from attentions import MultiHeadAttention

class Block(nn.Module):
	def __init__(self, d_model: int, expansion_factor: int = 4):
		super().__init__()
		self.MHA = MultiHeadAttention(d_model=d_model, n_heads=8)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.feed_forward = nn.Sequential(
			nn.Linear(d_model, d_model*expansion_factor),
			nn.GELU(),
			nn.Linear(d_model*expansion_factor, d_model)
		)
		self.dropout = nn.Dropout(0.1)
  
	def forward(self, x):
		""" LayerNorm -> MHA -> Add -> LayerNorm -> Feed Forward -> Add """
  
		x = self.norm1(x) # Pre Norm
		attn = self.MHA(x) # Attention layer
		x = x + self.dropout(attn) # Residual layer
		x = self.norm2(x) # Norm
		ff = self.feed_forward(x) # Feed forward layer
		x = x + self.dropout(ff) # Residual layer
		return x

class GPT2(nn.Module):
	def __init__(self, d_model: int = 768, N_stack: int = 12, vocab_size: int = 50257):
		super().__init__()
		self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
		self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model)
		self.blocks = nn.ModuleList([Block(d_model=d_model) for _ in range(N_stack)])
		self.norm = nn.LayerNorm(d_model)
		self.linear = nn.Linear(d_model, vocab_size)
		self.softmax = nn.Softmax(dim=-1)
		
	def forward(self, x):
		""" Input -> Embedding + Pos. Encoding -> Block x N -> LayerNorm -> Linear -> Softmax """
		x = self.embedding(x)
		x = x + self.pos_enc(x)
		for block in self.blocks:
			x = block(x)
		x = self.norm(x)
		x = self.linear(x)
		x = self.softmax(x)
		return x

if __name__ == "__main__":
	device = torch.device("mps")
	model = GPT2().to(device)
	batch_size, seq_len = 8, 1024
	inpt = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
	with torch.no_grad():
		out = model(inpt)
	print(out.shape)