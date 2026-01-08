import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizers import BPE_Tokenizer
# from transformers.dataset import Transformer_DS

# Read the textfile
full_text = open("/Users/macbookpro/Desktop/Transformers/data/quotes.txt", "r").readlines()

# Train the tokenizer on our data
tok_epochs = 5000
tokenizer = BPE_Tokenizer(text=" ".join(full_text))
tokenizer.fit(tok_epochs)

# Define dataset
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels}

train_dataset = Transformer_DS(tokens=tokenizer.text_tokens, block_size=128)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn) # Inputs of shape [batch_size, block_size]

# Define the model
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model = ...
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

epochs = 10
for epoch in tqdm.tqdm(range(epochs), desc="Training the model"):
    model.train()
    for inpt, target in train_loader:
        inpt, target = inpt.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(inpt)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()