# Standard library imports
import uuid
from argparse import Namespace
from typing import AsyncGenerator, Optional

# Third-party imports
import bentoml
import fastapi
from annotated_types import Ge, Le
from typing_extensions import Annotated

# Initialize FastAPI application
openai_api_app = fastapi.FastAPI()

# Constants
MAX_MODEL_LEN = 8192
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a helpful and respectful assistant. Provide safe, unbiased, and accurate answers. 
                If a question is unclear or you don't know the answer, explain why instead of guessing."""

MODEL_ID = "casperhansen/llama-3.3-70b-instruct-awq"

# Define OpenAI-compatible API endpoints
OPENAI_ENDPOINTS = [
    ["/chat/completions", "create_chat_completion", ["POST"]],
    ["/completions", "create_completion", ["POST"]],
    ["/models", "show_available_models", ["GET"]],
]


@bentoml.mount_asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    name="llama3.3-70b-instruct-awq",
    traffic={
        "timeout": 1200,
        "concurrency": 256,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb",
    },
)
class BentoVLLM:
    def __init__(self) -> None:
        """Initialize the BentoVLLM service with VLLM engine and tokenizer."""
        import vllm.entrypoints.openai.api_server as vllm_api_server
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        # Configure VLLM engine arguments
        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            max_model_len=MAX_MODEL_LEN,
            enable_prefix_caching=True,
        )

        # Initialize engine and tokenizer
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # Register API endpoints
        for route, endpoint_name, methods in OPENAI_ENDPOINTS:
            endpoint_func = getattr(vllm_api_server, endpoint_name)
            openai_api_app.add_api_route(
                path=route,
                endpoint=endpoint_func,
                methods=methods,
            )

        # Configure model arguments
        model_config = self.engine.engine.get_model_config()
        args = Namespace(
            model=MODEL_ID,
            disable_log_requests=True,
            max_log_len=1000,
            response_role="assistant",
            served_model_name=None,
            chat_template=None,
            lora_modules=None,
            prompt_adapters=None,
            request_logger=None,
            disable_log_stats=True,
            return_tokens_as_token_ids=False,
            enable_tool_call_parser=True,
            enable_auto_tool_choice=True,
            tool_call_parser="llama3_json",
            enable_prompt_tokens_details=False,
        )

        # Initialize application state
        vllm_api_server.init_app_state(
            self.engine, model_config, openai_api_app.state, args
        )

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Describe the process of photosynthesis in simple terms",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text based on the input prompt using the VLLM engine.

        Args:
            prompt: The user's input prompt
            system_prompt: Optional system prompt to guide the model's behavior
            max_tokens: Maximum number of tokens to generate

        Returns:
            AsyncGenerator yielding generated text chunks
        """
        from vllm import SamplingParams

        # Configure sampling parameters
        SAMPLING_PARAM = SamplingParams(
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )

        # Use default system prompt if none provided
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        # Prepare messages for chat
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate response stream
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

        # Stream the generated text
        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
