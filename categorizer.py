import ollama
import os

model = "llama3.2:1B"

input_file = "/media/aumoza/Strg_1/ollama-finetune/ollamaFreeCodeCamp/grocery_list.txt"
output_file = "/media/aumoza/Strg_1/ollama-finetune/ollamaFreeCodeCamp/categorized_grocery_list.txt"

# Check if the input file exists or not.
if not os.path.exists(input_file):
    print(f"Input file does not exists.")
    exit(1)

with open(input_file,"r") as f :
    items = f.read().strip()

prompt = f"""
You are an assistant that categorizes and sorts grocery items.
Here is a list of grocery items :
{items}

please
1. Categorize the items into appropriate categories.
2. Sort the items alphabetically within each category.
3. Present the list in clear and organized manner.
"""

# Send the prompt and get the response :
try:
    response = ollama.generate(model=model,prompt=prompt)
    generated_text = response.get("response","")

    with open(output_file,"w") as f:
        f.write(generated_text.strip())

        print(f"Categorized grocery list has been saved to {output_file}")
except Exception as e:
    print("An error occured: ",str(e))