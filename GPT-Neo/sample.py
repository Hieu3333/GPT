from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load the smallest GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"  # Smallest model
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

import torch

# Define the prompt
prompt = "What do you think about dog?"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Create attention mask if needed (e.g., if you have padding tokens)
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

# Generate with attention mask
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    num_return_sequences=1,
    do_sample=True
)


# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
