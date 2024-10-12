import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import fitz  # PyMuPDF for reading PDF
import os  # For working with file paths

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

def read_pdf(file_path):
    """Reads a PDF file and returns the text content."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_into_chunks(text, max_length=2000):
    """Splits the text into smaller chunks."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def ask_question(question, context, conversation_history):
    """Uses the model to generate an answer based on the question, context, and conversation history."""
    # Add the previous conversation history to maintain context
    conversation_history += f"\nQuestion: {question}"
    input_text = f"Context: {context}\n{conversation_history}"

    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate the output with a limited number of new tokens
    output_sequences = model.generate(inputs, max_new_tokens=300, num_return_sequences=1)

    answer = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    # Update conversation history with the model's response
    conversation_history += f"\nAnswer: {answer}"
    
    return answer, conversation_history

def read_multiple_pdfs(pdf_directory):
    """Reads all PDF files from a directory and combines their content."""
    combined_text = ""
    for file_name in os.listdir(pdf_directory):
        if file_name.endswith('.pdf'):  # Only process PDF files
            file_path = os.path.join(pdf_directory, file_name)
            combined_text += read_pdf(file_path) + "\n"  # Combine content with a newline separator
            print(f"Processed {file_name}")
    return combined_text

# Path to your PDF directory containing multiple PDFs
pdf_directory = 'docs'  # Replace with your folder containing multiple PDFs

# Read the combined content of all PDFs in the directory
pdf_content = read_multiple_pdfs(pdf_directory)

# Split the combined PDF content into manageable chunks
context_chunks = split_into_chunks(pdf_content, max_length=2000)

# Initialize the conversation history
conversation_history = ""

# Interactive loop for conversation
if context_chunks:
    print("The conversation has started. Ask your questions based on the PDF content.")

    first_chunk = context_chunks[0]  # Start with the first chunk
    context = first_chunk  # You can later implement dynamic context switching
    
    while True:
        # Ask user for input (question)
        question = input("\nYour question: ")

        if question.lower() in ['exit', 'quit', 'q']:  # Exit condition
            print("Ending the conversation. Goodbye!")
            break

        # Ask the question and get the answer
        answer, conversation_history = ask_question(question, context, conversation_history)

        # Display the answer
        print(f"\nModel's answer: {answer}")
else:
    print("The PDFs are empty or could not be read.")
