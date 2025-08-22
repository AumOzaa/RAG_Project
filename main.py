import requests
import json

url = "http://localhost:11434/api/generate"

data = {
    "model": "llama3.2:latest",
    "prompt": "Who was Steve Jobs and what lessons can we take from him"
}

response = requests.post(url, json=data, stream=True)

# Check the response status
if response.status_code == 200:
    print("Generated Text : ", end="", flush=True)
    # Iterate over the streaming response
    for line in response.iter_lines():
        if line:  # Skip keep-alive new lines
            decode_line = line.decode("utf-8")
            result = json.loads(decode_line)
            # Get text from the response
            generate_text = result.get("response", "")
            print(generate_text, end="", flush=True)
else:
    print("Error:", response.status_code, response.text)
