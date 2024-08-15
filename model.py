import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from utils import Loading


load_dotenv()

# Constants
TEMPERATURE = 0.5
TOKEN_LIMIT = 8192

class ModelOptions: 
    def __init__(self, 
                 temperature=0, 
                 token_limit=4096, 
                 json_mode=False) -> None:
        self.temperature = temperature
        self.token_limit = token_limit
        self.json_mode = json_mode
    
    def to_json(self) -> dict:
        return {
            "temperature": self.temperature,
            "token_limit": self.token_limit,
            "json_mode": self.json_mode
        }
    
class Model:
    SUPPORTED_MODEL_PROVIDERS = {'openai', 'anthropic'}
    MODEL_MAPPING = {}

    def __init__(self, model_provider: str, model_name: str, base_model_options: ModelOptions = ModelOptions()):
        """
        Initialize the Model class.

        :param model_provider: The provider of the model (e.g. openai).
        :param model_name: The name of the model (e.g. gpt-4o).
        :param options: Optional parameters for the model.
        """

        if model_provider not in self.SUPPORTED_MODEL_PROVIDERS:
            raise ModelError(f'Unsupported model provider: {self.model_provider}')
        self.model_provider = model_provider
        self.model_name = model_name

        self.id = model_provider + "-" + model_name
        self.client = None
        self.base_model_options = base_model_options

        Model.MODEL_MAPPING[self.id] = self

    def load_model(self) -> None:
        match self.model_provider:
            case "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is not set")
                print(f"Initializing OpenAI client with API key: {api_key[:5]}...")  # Print first 5 chars for debugging
                self.client = OpenAI(api_key=api_key)
            case "anthropic":
                self.client = Anthropic()

    def get_embedding(self, text, model="text-embedding-ada-002"):
        """Generates embedding for the given text using OpenAI."""
        try:
            response = self.client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
        except Exception as ex:
            print(f"Failed to generate embedding: {ex}")
            return None

    def query(self, system_message: str, user_prompt: str, query_model_options: ModelOptions = None) -> str:
        if query_model_options is None:
            model_options = self.base_model_options
        else:
            model_options = query_model_options
        loading = Loading(loading_message=f"Querying {self.model_name}...")
        match self.model_provider:
            case "openai":
                messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_prompt}]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"} if model_options.json_mode else None,
                    temperature=model_options.temperature,
                    max_tokens=model_options.token_limit,
                )
                if model_options.json_mode:
                    output = json.loads(response.choices[0].message.content)
                else:
                    output = response.choices[0].message.content
            case "anthropic":
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                ]
                if model_options.token_limit > 4096:
                    response = self.client.messages.create(
                        system=system_message,
                        max_tokens=model_options.token_limit,
                        messages=messages,
                        model=self.model_name,
                        temperature=model_options.temperature,
                        extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                    )
                else:
                    response = self.client.messages.create(
                        system=system_message,
                        max_tokens=model_options.token_limit,
                        messages=messages,
                        model=self.model_name,
                        temperature=model_options.temperature,
                    )
                if (
                    model_options.json_mode
                ):  # https://github.com/anthropics/anthropic-cookbook/blob/main/misc/how_to_enable_json_mode.ipynb

                    def extract_json(response):
                        json_start = response.index("{")
                        json_end = response.rfind("}")
                        return response[json_start : json_end + 1]

                    output = json.loads(extract_json(response.content[0].text))
                else:
                    output = response.content[0].text
        loading.stop_loading()
        return output

    def structured_query(
        self, system_message: str, user_prompt: str, response_format, query_model_options: ModelOptions = None
    ) -> str:
        assert self.model_provider == 'openai', 'Only OpenAI supports structured queries currently.'
        if query_model_options is None:
            model_options = self.base_model_options
        else:
            model_options = query_model_options
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_prompt}]
        loading = Loading(loading_message=f"Querying {self.model_name}...")
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=response_format,
            temperature=model_options.temperature,
            max_tokens=model_options.token_limit,
        )
        loading.stop_loading()
        message = completion.choices[0].message
        if message.parsed:
            print(message.parsed)
            return message.parsed
        else:
            print(message.refusal)
            return message.refusal

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Model):
            return False
        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self) -> str:
        return self.id


class ModelError(Exception):
    pass
