import torch

def topk_sampler(logits: torch.tensor, k: int, temp: float = 1.0):
    """ Top K Sampling implementation

    Args:
        logits (torch.tensor): The model output logits
        k (int): The number of top tokens to consider
        temp (float, optional): Temperature for scaling logits. Defaults to 1.0.
    """
    # 0. Divide logits by temp
    logits = logits / temp
    
    # 1. Top K candidates
    top_k, indices = torch.topk(logits, k, sorted=True)
    
    # 2. Compute probabilities
    probs = torch.nn.functional.softmax(top_k, dim=-1)
    
    # 3. Sample from our probs
    idx = torch.multinomial(probs, 1)
    
    return torch.gather(indices, index=idx, dim=-1) # Return the logit with correct index
    
if __name__ == "__main__":
    logits = torch.rand(4, 40)
    out = topk_sampler(logits=logits, k=4)
    print(out)