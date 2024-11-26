import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Paths to model and tokenizer
model_path = 'meta-llama'
tokenizer_path = 'meta-llama'

# Load tokenizer and set pad_token
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

# Configure and load model with int8 quantization
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, torch_dtype=torch.float16)

# Compile and move model to the correct device
model = torch.compile(model, mode='reduce-overhead')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Initialize conversation history
conversation_history = ""

def ask_question(question):
    """Generates an answer, maintaining conversation history."""
    global conversation_history

    # Add the current question to the conversation history
    conversation_history += f"\nUser: {question}\nAI:"
    input_text = conversation_history

    # Tokenize and move to device
    inputs = tokenizer(
        input_text, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=2048
    )
    inputs['attention_mask'] = inputs['input_ids'] != tokenizer.pad_token_id  # Explicit attention mask
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate output
    generated_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],  # Explicit mask to avoid warnings
        max_new_tokens=800,
        num_return_sequences=1,
        num_beams=3,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,  # Enable sampling for diverse output
    )

    # Decode the response
    decoded_tokens = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    new_response = decoded_tokens.split("AI:")[-1].strip()  # Get the latest response

    # Print the response all at once
    print(f"AI: {new_response}")
    
    # Update conversation history
    conversation_history += f" {new_response}"

    # Return the full response
    return new_response

# Example conversational usage
while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        print("Ending conversation.")
        break
    
    ask_question(question)
