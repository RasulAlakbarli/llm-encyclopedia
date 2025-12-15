import math
import torch
import torch.nn as nn

class GroupQueryAttention(nn.Module):
	"""
	Group Query Attention (GQA) implementation
	"""
	def __init__(self, d_model, Q_heads, KV_heads):
		super().__init__()
		assert d_model % Q_heads == 0, "d_model must be divisible by number of query heads."
		assert d_model % KV_heads == 0, "d_model must be divisible by number of key-value heads."
		assert Q_heads % KV_heads == 0, "Number of query heads must be divisible by number of key-value heads."
		
		self.Q_heads = Q_heads
		self.KV_heads = KV_heads
		self.d_q_head = d_model // Q_heads
		self.d_kv_head = self.d_q_head * KV_heads
		
		self.W_Q = nn.Linear(d_model, d_model)
		self.W_K = nn.Linear(d_model, self.d_kv_head)
		self.W_V = nn.Linear(d_model, self.d_kv_head)
		
		self.linear = nn.Linear(d_model, d_model)
		self.softmax = nn.Softmax(dim=-1)
		self.dropout = nn.Dropout(0.2)
		
	def forward(self, x, mask=None):
		Q = self.W_Q(x)
		K = self.W_K(x)
		V = self.W_V(x)
		B, N, d_model = Q.size()
		
		Q = Q.view(B, N, self.Q_heads, self.d_q_head).transpose(1, 2) # [B, N_q_head, N, d_q_head]
		K = K.view(B, N, self.KV_heads, self.d_q_head) # [B, N, N_kv_head, d_q_head]
		V = V.view(B, N, self.KV_heads, self.d_q_head) # [B, N, N_kv_head, d_q_head]
		
		K = K.repeat_interleave(self.Q_heads // self.KV_heads, dim=2).transpose(1, 2)
		V = V.repeat_interleave(self.Q_heads // self.KV_heads, dim=2).transpose(1, 2)
		
		attn = Q@K.transpose(-2, -1)/math.sqrt(Q.size(-1))
		if mask is not None:
			attn = attn.masked_fill(mask == 0, float("-1e9"))

		scores = self.softmax(attn)
  
		x = self.dropout(scores)
		x = x@V # (batch_size, n_heads, seq_len, d_head)
		x = x.transpose(1, 2).contiguous().view(B, N, d_model)

		return self.linear(x)
		 
        
if __name__ == "__main__":
	d_model, Q_heads, KV_heads = 512, 8, 2
	gqa = GroupQueryAttention(d_model, Q_heads, KV_heads)
	inpt = torch.rand(1, 16, d_model)
	with torch.no_grad():
		out = gqa(inpt)
	print(out.shape)