import os

from openai import OpenAI

BENTOML_API_KEY = os.getenv("BENTO_CLOUD_API_KEY")  #
if BENTOML_API_KEY is None:
    raise ValueError("BENTOML_API_KEY environment variable is not set.")


client = OpenAI(
    base_url="https://llama-3-3-70-b-instruct-awq-zkd2-39800880.mt-guc1.bentoml.ai/v1",
    api_key=BENTOML_API_KEY,
)

chat_completion = client.chat.completions.create(
    model="casperhansen/llama-3.3-70b-instruct-awq",
    messages=[
        {"role": "user", "content": "What is a black hole and how does it work?"}
    ],
    stream=True,
    stop=["<|eot_id|>", "<|end_of_text|>"],
)
for chunk in chat_completion:
    print(chunk.choices[0].delta.content or "", end="")
