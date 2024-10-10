import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths to your model and tokenizer files
model_path = 'meta-llama'  # e.g., 'path/to/your/model/file.pth' or 'path/to/your/model/file.safetensors'
tokenizer_path = 'meta-llama'  # e.g., 'path/to/your/tokenizer'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)  # Use torch.float32 if not using GPU

# Move the model to the appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Initialize conversation history
conversation_history = ""

# Start a loop for the conversation
while True:
    # Get user input
    user_input = input("You: ")

    # Exit the conversation loop if the user types 'exit'
    if user_input.lower() == 'exit':
        print("Conversation ended.")
        break

    # Append user input to the conversation history
    conversation_history += f"User: {user_input}\n"

    # Tokenize the conversation history
    inputs = tokenizer.encode(conversation_history, return_tensors='pt').to(device)

    # Generate the model's response
    output_sequences = model.generate(inputs, max_new_tokens=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode the model's response
    model_response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    # Extract the response (since history is included, split by user input)
    model_response = model_response[len(conversation_history):]

    # Add the model's response to the conversation history
    conversation_history += f"Model: {model_response}\n"

    # Print the model's response
    print(f"Model: {model_response}")
