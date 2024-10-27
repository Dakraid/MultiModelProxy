import re
from typing import LiteralString

from mistralai import Mistral
from openai import OpenAI

from src.config import Config


def text_to_chat_completion(
    config: Config, prompt: str
) -> list[
    dict[str, LiteralString | str]
    | dict[str, LiteralString | str]
    | dict[str, LiteralString | str]
]:
    messages = []
    pattern = config.configuration.regex.text_to_chat_pattern
    matches = re.finditer(pattern, prompt, re.MULTILINE | re.DOTALL)

    for matchNum, match in enumerate(matches):
        groups = match.groupdict()
        if groups["System"]:
            messages.append(
                {
                    "role": "system",
                    "content": groups["System"]
                    .replace("[INST]", "")
                    .replace("[/INST] Understood.</s>", "")
                    .strip(),
                }
            )
        elif groups["User"]:
            messages.append(
                {"role": "user", "content": groups["User"].replace("\\", "").strip()}
            )
        elif groups["Assistant"]:
            messages.append(
                {
                    "role": "assistant",
                    "content": groups["Assistant"].replace("\\", "").strip(),
                }
            )

    return messages


class InferenceBase:
    def __init__(self, config: Config):
        self.config = config

    def completion(self, prompt: str):
        pass


class TabbyApiInference(InferenceBase):
    def __init__(self, config: Config):
        super().__init__(config)

    def completion(self, prompt: str):
        pass


class MistralInference(InferenceBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.client = Mistral(api_key=config.configuration.inference.mistral.api_key)

    def completion(self, prompt: str):
        chat = text_to_chat_completion(self.config, prompt)

        username = self.config.configuration.general.default_username
        user_regex = re.compile(self.config.configuration.regex.username_pattern)
        user_match = user_regex.findall(prompt, re.MULTILINE)
        if user_match[0]:
            username = user_match[0]

        end_index = prompt.rfind("]")
        end_index2 = prompt.rfind("[")
        character_raw = prompt[end_index + 1 :]
        character = character_raw.replace(":", "").strip()

        chat.append(
            {
                "role": "user",
                "content": self.config.configuration.general.prompt.format(
                    username=username, character=character
                ),
            }
        )
        chat_response = self.client.chat.complete(
            model=self.config.configuration.inference.mistral.model, messages=chat
        )

        spacing = "[/INST] "
        if prompt[end_index2 - 1 : end_index2] != ">":
            spacing = ""
        message = chat_response.choices[0].message.content
        expanded = (
            prompt[:end_index2]
            + spacing
            + "Thoughts: "
            + message
            + "</s>[/INST]"
            + character_raw
        )
        return message, expanded


class OpenRouterInference(InferenceBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.configuration.inference.openrouter.api_key,
        )

    def completion(self, prompt: str):
        chat = text_to_chat_completion(self.config, prompt)

        username = self.config.configuration.general.default_username
        user_regex = re.compile(self.config.configuration.regex.username_pattern)
        user_match = user_regex.findall(prompt, re.MULTILINE)
        if user_match[0]:
            username = user_match[0]

        end_index = prompt.rfind("]")
        end_index2 = prompt.rfind("[")
        character_raw = prompt[end_index + 1 :]
        character = character_raw.replace(":", "").strip()

        chat.append(
            {
                "role": "user",
                "content": self.config.configuration.general.prompt.format(
                    username=username, character=character
                ),
            }
        )

        chat_response = self.client.chat.completions.create(
            model=self.config.configuration.inference.openrouter.model, messages=chat
        )

        spacing = "[/INST] "
        if prompt[end_index2 - 1 : end_index2] != ">":
            spacing = ""
        message = chat_response.choices[0].message.content
        expanded = (
            prompt[:end_index2]
            + spacing
            + "Thoughts: "
            + message
            + "</s>[/INST]"
            + character_raw
        )
        return message, expanded
