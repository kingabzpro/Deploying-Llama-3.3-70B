# Deploying-Llama-3.3-70B
Deploy and serve Llama 3.3 70B with AWQ quantization using vLLM and BentoML.

## Clone the repository
```bash
git clone https://github.com/kingabzpro/Deploying-Llama-3.3-70B.git
cd Deploying-Llama-3.3-70B.git
```

## Deployment

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Logged in to BentoCloud:
```bash
bentoml cloud login
```
3. Deploy the model:
```bash
bentoml deploy .
```
## Inference

The model can be accessed via CURL command, BentoML Python Client, or OpenAI Python client.:

```python
from openai import OpenAI

client = OpenAI(base_url="<BentoCloud endpoint>", api_key="<Your BentoCloud API key>")

chat_completion = client.chat.completions.create(
    model="casperhansen/llama-3.3-70b-instruct-awq",
    messages=[
        {
            "role": "user",
            "content": "What is a black hole and how does it work?"
        }
    ],
    stream=True,
	stop=["<|eot_id|>", "<|end_of_text|>"],
)
for chunk in chat_completion:
    print(chunk.choices[0].delta.content or "", end="")
```