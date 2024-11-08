from openai import OpenAI

from src import Config, InferenceBase
from src.utility import database


class OpenRouterInference(InferenceBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.configuration.inference.openrouter.api_key,
        )

    async def cot_completion(self, completion_request: dict, stored_last_message: str) -> tuple[str | None, str, str]:
        self.setup_vars(completion_request)
        chat, last_message = self.prepare_chat_completion()

        if stored_last_message != last_message or self.new_cot_prompt:
            while self.response == "" or self.response_tokens < 200:
                chat_response = self.client.chat.completions.create(
                    model=self.config.configuration.inference.openrouter.model,
                    messages=chat,
                )
                self.response = chat_response.choices[0].message.content
                self.response_tokens = chat_response.usage.completion_tokens
            self.completion = self.complete_chat_completion()
            await database.insert_log(self.response, self.response_tokens)

        return self.response, self.completion, last_message
