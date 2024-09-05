from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model and tokenizer from saved directory
model_name = "gpt2"
# model_name = "./fine_tuned_model/"  # Path to your fine-tuned model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the pad_token_id to the eos_token_id to fix the padding issue
model.config.pad_token_id = model.config.eos_token_id

def generate_philosophical_response(prompt):
    # Add an introductory phrase to set the tone
    prompt = "As a philosopher, I ponder: " + prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        top_k=60,
        do_sample=True)
    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    return response

def chat_with_philosopher():
    print("Welcome to the philosophical chatbot. Ask me anything!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Reflect well.")
            break
        response = generate_philosophical_response(user_input)
        print("Philosopher Bot: " + response)

# Start the conversation
chat_with_philosopher()
