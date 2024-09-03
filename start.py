from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can start with GPT-2 and later switch to GPT-3/4
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2,
                             top_p=0.95, top_k=60)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the function with a philosophical question
user_input = "What is the meaning of life?"
print(generate_response(user_input))
