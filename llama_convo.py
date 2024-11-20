import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    """Generates an answer to a single question, maintaining conversation history."""
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
        max_length=512
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate output
    output_sequences = model.generate(
        inputs['input_ids'],
        max_new_tokens=300,     #token depends on the generate text
        num_return_sequences=1,
        num_beams=3,  # Beam search for better quality
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the response
    answer = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    # Append only the new response to the conversation history
    new_response = answer.split("AI:")[-1].strip()
    conversation_history += f" {new_response}"
    
    # Return only the response
    return new_response

# Example conversational usage
while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        print("Ending conversation.")
        break
    
    answer = ask_question(question)
    print(answer)
