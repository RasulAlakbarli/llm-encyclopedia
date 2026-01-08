# Transformers: From Scratch to Understanding

A learning repository exploring the foundational concepts of Large Language Models (LLMs), starting with the original Transformer architecture.

## Topics Covered

### [Tokenization](./tokenizers/)
- [BPE Tokenizer](./tokenizers/bpe_tokenizer.py)

### [Positional Encoding](./positional_encoders/)
- [Sinusoidal Positional Encoding](./positional_encoders/sinusoidal_positional_encoding.py)
- [Rotary Positional Encoding (RoPE)](./positional_encoders/rotary_positional_encoding.py)

### [Attention Mechanisms](./attentions/)
- [Scaled Dot-Product Attention](./attentions/scaled_dot_product_attention.py)
- [Multi-Head Attention](./attentions/multi_head_attention.py)
- [Group Query Attention (GQA)](./attentions/group_query_attention.py)

### [Models](./models/)
- [Transformer](./models/transformer.py)
- [GPT-2](./models/gpt2.py)
- [Mixture of Experts (MoE)](./models/moe.py)

### [Sampling Strategies](./samplers/)
- [Top-K Sampling](./samplers/top_k.py)
- [Top-P (Nucleus) Sampling](./samplers/top_p.py)

### [Training](./training/)
- [Training Scripts](./training/train.py)

## Blog

For detailed explanations with visuals, check out the accompanying blog posts:

- [Coming Soon] - <Blog link>

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Paper
- [Andrej Karpathy's Github](https://github.com/karpathy) - Check minGPT, nanoGPT, minbpe repos
- [Andrej Karpathy's YouTube](https://www.youtube.com/@AndrejKarpathy) - Check videos about GPT tokenizer and GPT model
- [Medium Post by Sumith Madupu](https://medium.com/@sumith.madupu123/understanding-transformer-architecture-using-simple-math-be6c2e1cdcc7) - A great post explaining transformer architecture with simple math.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 Paper
- [Gemini Pro](https://gemini.google.com/) - Very helpful for understanding the concepts and math behind LLMs.