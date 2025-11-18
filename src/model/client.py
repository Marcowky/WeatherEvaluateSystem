from openai import OpenAI

from util.config import load_config


class ModelClient:
    def __init__(self, api_type: str = None):
        self.config = self.load_api_config(api_type=api_type)
        self.client = OpenAI(
            api_key=self.config["api_key"], 
            base_url=self.config["base_url"]
        )

    def load_api_config(self, api_type: str = None) -> dict:
        config = load_config()
        if api_type is None:
            api_type = config["llm_api"]["default"]
        api_config = config["llm_api"][api_type]
        return api_config

    def chat_with_messages(self, model: str, messages: list, **kwargs) -> dict:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response
    
    def chat_with_messages_return_text(self, model: str, messages: list, **kwargs) -> str:
        response = self.chat_with_messages(model=model, messages=messages, **kwargs)
        return response.choices[0].message.content

    def chat_with_prompt(self, model: str, prompt: str, **kwargs) -> dict:
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_with_messages(model=model, messages=messages, **kwargs)
        return response
    
    def chat_with_prompt_return_text(self, model: str, prompt: str, **kwargs) -> str:
        response = self.chat_with_prompt(model=model, prompt=prompt, **kwargs)
        return response.choices[0].message.content