curl -s -X POST \
    'https://llama-3-3-70-b-instruct-awq-zkd2-39800880.mt-guc1.bentoml.ai/generate' \
    -H "Authorization: Bearer $(echo $BENTO_CLOUD_API_KEY)" \
    -H 'Content-Type: application/json' \
    -d '{
        "max_tokens": 1024,
        "prompt": "Describe the process of photosynthesis in simple terms"
    }' 