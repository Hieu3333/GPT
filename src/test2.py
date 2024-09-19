import tiktoken

# Load the GPT-2 tokenizer
encoder = tiktoken.get_encoding("gpt2")

# Get the vocabulary (mapping from token to index)
print(encoder.eot_token)

# # Get the tokens and their IDs
# tokens = list(token_to_id.keys())
# values = list(token_to_id.values())

# # Print tokens and their IDs
# with open('list_tok.txt','r') as file:
#     for token, value in zip(tokens, values):
#         file.write(f"Token: {token}, ID: {value}")
