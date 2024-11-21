# LLM (Large Language Model) - META_LLAMA

# Install Requirements
```
pip install -r requirements.txt
```
# Download Pre-Trained Models 
```
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main
```
# How to RUN

Download the all files from [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main) (There are other lightweight files)

Make a file and put those downloaded files into there.

llama_convo.py is for normal conversation and llama_convo_doc.py is to ask pdf related question.

Change or adjust num_beam for better resluts. 

# llama_convo

It runs normally like other online LLM with no limitation.

# llama_convo_doc

It is made to read pdf or documents and make summarise about them.

# food_llama

Making LLM to talk about only the specific type of related topic (food) and reject when user wants to talk about others things.

# Pros of Higher Beam Size

1. Improved Quality:
A larger beam size can find sequences with higher overall probabilities, leading to more coherent responses.
2. Diversity:
More beams mean the model can explore a broader range of possible outputs.
3. Avoiding Traps:
With fewer beams, the model might prematurely settle on a suboptimal sequence.

# Cons of Higher Beam Size

1. Increased Computation:
Higher beam sizes require more computation, which increases latency and resource usage.
2. Reduced Diversity:
If the beams converge to similar high-probability sequences, increasing the beam size won’t add value.
3. Over-optimization:
In some cases, higher beam sizes may cause the model to favor overly generic or "safe" responses.
4. Diminishing Returns:
Beyond a certain point (e.g., num_beams=5 or num_beams=10), quality improvements tend to plateau or even degrade.

Run Python file from Terminal to start. 

gpu_test.py is to make sure graphic card (cuda) is using. 

# Hardware Usage 

![Screenshot 2024-10-10 034412](https://github.com/user-attachments/assets/a29809f4-ad12-4dee-a5c1-e93394d259c3)

# My PC Specs

> CPU  - Ryzen 9 5900HS (8 Cores 16 Threads)

> RAM  - 24GB (3200MHz)

> GPU  - RTX 3060 (VRAM - 6GB)
