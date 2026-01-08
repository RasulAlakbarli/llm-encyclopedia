import math
import torch

def topp_sampler(logits: torch.tensor, p: float, temp: float = 1.0):
    """Top P Sampling (Nucleus Sampling) implementation

    Args:
        logits (torch.tensor): The model output logits
        p (float): The cumulative probability threshold
        temp (float, optional): Temperature for scaling logits. Defaults to 1.0.
    """
    # 0. Divide logits by temp
    logits = logits / temp
    
    # 1. Sort logits
    sorted_logits, indices = torch.sort(logits, dim=-1, descending=True)
    
    # 2. Compute probabilities
    probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
    
    # 3. Compute cumulative sum
    probs_cum = torch.cumsum(probs, dim=-1)

    # 4. Create the mask 
    mask = probs_cum > p
    # Shift the mask to the right
    mask = torch.roll(mask, 1, dims=-1)
    mask[:, 0] = False
    
    # 5. Mask the probabiliyies
    logits = sorted_logits.masked_fill(mask, -float("inf"))
    
    # 6. Re normalize with softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # 7. Sample from probs
    idx = torch.multinomial(probs, 1)

    return indices.gather(index=idx, dim=-1)
    
if __name__ == "__main__":
    logits = torch.rand(2, 9)
    out = topp_sampler(logits=logits, p=0.7)
    print(out)