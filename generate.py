import torch
from model import GPT, GPTConfig
from transformers import GPT2TokenizerFast

def generate(prompt, max_new_tokens=50, temperature=1.0):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokens = tokenizer(prompt, return_tensors='pt').input_ids
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
    )
    model = GPT(config)
    checkpoint = torch.load('checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = tokens[:, -config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

if __name__ == '__main__':
    while True:
        prompt = input("You: ")
        print("Bot:", generate(prompt))
