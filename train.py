from datasets import load_dataset, get_dataset_split_names
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load JSON dataset
dataset_dict = load_dataset("tylercross/platos_socrates")
dataset = dataset_dict["train"]
# Create a test split that is 10% of the original dataset
split_dataset = dataset.train_test_split(test_size=0.1)


# Load the GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure the model and tokenizer are padded correctly
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    inputs = examples["input"]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    # Labels are the same as input_ids in a language modeling task
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Tokenize the dataset
tokenized_dataset = split_dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',         # Output directory for model and logs
    overwrite_output_dir=True,      # Overwrite the output directory if it exists
    num_train_epochs=3,             # Number of training epochs
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,   # Batch size for evaluation
    warmup_steps=500,               # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,              # Strength of weight decay
    logging_dir='./logs',           # Directory for storing logs
    logging_steps=10,
    eval_strategy="epoch",          # Evaluate after each epoch
    save_strategy="epoch"           # Save model after each epoch
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test']
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
