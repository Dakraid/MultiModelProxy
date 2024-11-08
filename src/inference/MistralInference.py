import json

from loguru import logger
from mistralai import Mistral
from mistralai.types.basemodel import Unset

from src import Config, InferenceBase
from src.utility import database


class MistralInference(InferenceBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.completion = None
        self.response_tokens = None
        self.response = None
        self.client = Mistral(api_key=config.configuration.inference.mistral.api_key)

    async def process_prompt(self, chat: list[dict[str, str]], prompt: dict[str, str], prefix: str) -> tuple[str, int]:
        key, value = next(iter(prompt.items()))
        constructed_prompt = f"{prefix}\n# {key}\n{value}"
        logger.info(f"Generating answer for {key} section.")
        chat.append(
            {"role": "user", "content": constructed_prompt.format(username=self.username, character=self.character)}
        )
        while True:
            chat_response = self.client.chat.complete(
                model=self.config.configuration.inference.mistral.model, messages=chat
            )
            response = chat_response.choices[0].message.content
            response_tokens = chat_response.usage.completion_tokens
            if response_tokens >= 20:
                chat.pop()
                if f"# {key}" not in response:
                    response = f"# {key}\n{response}\n"
                logger.info(f"Generation for {key} section completed.")
                return response, response_tokens

    async def cot_completion(
        self, completion_request: dict, stored_last_message: str
    ) -> tuple[str | None | Unset, str, str]:
        self.setup_vars(completion_request)
        chat, last_message = self.prepare_chat_completion()

        if "!JSON" in self.cot_prompt:
            logger.info("JSON mode detected.")
            prompts_raw = self.cot_prompt.replace("!JSON", "")
            prompts_json = json.loads(prompts_raw)
            prompts = prompts_json["prompts"]
            logger.info(f"Found {len(prompts)} sections.")

            responses: list[tuple[str, int]] = []
            for prompt in prompts:
                responses.append(await self.process_prompt(chat, prompt, prompts_json["prefix"]))

            self.response = "\n".join([r[0] for r in responses])
            self.response = self.response.replace("\n\n", "\n")
            self.response_tokens = sum([r[1] for r in responses])
            self.completion = self.complete_chat_completion()
            await database.insert_log(self.response, self.response_tokens)
        else:
            logger.info("Default mode detected.")
            chat.append(
                {"role": "user", "content": self.cot_prompt.format(username=self.username, character=self.character)}
            )

            if stored_last_message != last_message or self.new_cot_prompt:
                logger.info("New context, generating new CoT.")
                while self.response == "" or self.response_tokens < 200:
                    chat_response = self.client.chat.complete(
                        model=self.config.configuration.inference.mistral.model, messages=chat
                    )
                    self.response = chat_response.choices[0].message.content
                    self.response_tokens = chat_response.usage.completion_tokens
                self.completion = self.complete_chat_completion()
                await database.insert_log(self.response, self.response_tokens)
            else:
                logger.info("No changes from last request, returning previous saved CoT.")

        return self.response, self.completion, last_message
