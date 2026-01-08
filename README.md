# Transformers: From Scratch to Understanding

A learning repository exploring the foundational concepts of Large Language Models (LLMs).

## Topics Covered

### Tokenization
- [BPE Tokenizer](./tokenizers/bpe_tokenizer.py)

### Positional Encoding
- [Sinusoidal Positional Encoding](./positional_encoders/sinusoidal_positional_encoding.py)
- [Rotary Positional Encoding (RoPE)](./positional_encoders/rotary_positional_encoding.py)

### Attention Mechanisms
- [Scaled Dot-Product Attention](./attentions/scaled_dot_product_attention.py)
- [Multi-Head Attention](./attentions/multi_head_attention.py)
- [Group Query Attention (GQA)](./attentions/group_query_attention.py)

### Models
- [Transformer](./models/transformer.py)
- [GPT-2](./models/gpt2.py)
- [Mixture of Experts (MoE)](./models/moe.py)

### Sampling Strategies
- [Top-K Sampling](./samplers/top_k.py)
- [Top-P (Nucleus) Sampling](./samplers/top_p.py)

### Training
- [Training Scripts](./training/train.py)

## Blogs

For detailed explanations with visuals, check out the accompanying blog posts:

- [Notes from Ultrascale Playbook](https://curly-answer-0de.notion.site/Notes-from-Ultrascale-Playbook-2d21509c2e9e80a0a371cda05b25903c) - Distilled knowledge about distributed training strategies for large models from the book "Ultrascale Playbook" by HuggingFace.

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Paper
- [Andrej Karpathy's Github](https://github.com/karpathy) - Check minGPT, nanoGPT, minbpe repos
- [Andrej Karpathy's YouTube](https://www.youtube.com/@AndrejKarpathy) - Check videos about GPT tokenizer and GPT model
- [Medium Post by Sumith Madupu](https://medium.com/@sumith.madupu123/understanding-transformer-architecture-using-simple-math-be6c2e1cdcc7) - A great post explaining transformer architecture with simple math.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 Paper
- [Gemini Pro](https://gemini.google.com/) - Very helpful for understanding the concepts and math behind LLMs.