import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
	"""
	Scaled Dot-Product Attention
	"""
	def __init__(self, d_attn):
		super().__init__()
		self.d_attn = d_attn
  
		self.W_Q = nn.Linear(self.d_attn, self.d_attn)
		self.W_K = nn.Linear(self.d_attn, self.d_attn)
		self.W_V = nn.Linear(self.d_attn, self.d_attn)
		self.softmax = nn.Softmax(dim=-1)
	def forward(self, x, mask=None):
		Q = self.W_Q(x)
		K = self.W_K(x)
		V = self.W_V(x)
		attn = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(Q.size(-1))
		if mask is not None:
			attn = attn.masked_fill(mask == 0, float("-1e9"))
		scores = self.softmax(attn)
  
		return torch.matmul(scores, V)