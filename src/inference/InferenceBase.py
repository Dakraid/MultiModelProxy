import re
from functools import lru_cache
from typing import Dict, List, Tuple

from src import Config


class InferenceBase:
    def __init__(self, config: Config):
        self.prompt = None
        self.completion_request = None
        self.config = config
        self.st_enabled = config.configuration.general.st_extension.enabled
        self._reset_state()

    def _reset_state(self) -> None:
        self.response = ""
        self.response_tokens = 0
        self.completion = ""
        self.completion_request = {}
        self.prompt = ""
        self.cot_prompt = ""
        self.pattern = ""
        self.username = ""
        self.character = ""
        self.character_raw = ""
        self.first_index = 0
        self.last_index = 0
        self.new_cot_prompt = False

    @lru_cache(maxsize=128)
    def _compile_regex(self, pattern: str) -> re.Pattern:
        return re.compile(pattern, re.MULTILINE | re.DOTALL)

    def full_completion(self, completion_request: dict, stored_last_message: str):
        pass

    def setup_vars(self, completion_request: dict) -> None:
        self._reset_state()
        self.completion_request = completion_request
        self.prompt = completion_request.get("prompt", "")

        if self.st_enabled:
            self._setup_st_vars(completion_request)
        else:
            self._setup_non_st_vars()

    def _setup_st_vars(self, completion_request: dict) -> None:
        self.username = completion_request.get("username", self.config.configuration.general.default_username)
        self.character = completion_request.get("character", "Character")
        self.character_raw = f" {self.character}:"
        self.pattern = self.config.configuration.general.st_extension.text_to_chat_pattern.format(
            username=re.escape(self.username), character=re.escape(self.character)
        )
        self._setup_common_vars(completion_request)

    def _setup_non_st_vars(self) -> None:
        self.username = self.config.configuration.general.default_username
        user_regex = self._compile_regex(self.config.configuration.regex.username_pattern)
        if user_match := user_regex.findall(self.prompt):
            self.username = user_match[0]

        self.first_index = self.prompt.rfind("]")
        self.character_raw = self.prompt[self.first_index + 1 :]
        self.character = self.character_raw.replace(":", "").strip()
        self.pattern = self.config.configuration.regex.text_to_chat_pattern

    def _setup_common_vars(self, completion_request: dict) -> None:
        self.first_index = self.prompt.rfind("]")
        self.last_index = self.prompt.rfind("[")
        current_cot_prompt = completion_request.get("cot_prompt", self.config.configuration.general.prompt)
        self.new_cot_prompt = self.cot_prompt == current_cot_prompt
        self.cot_prompt = current_cot_prompt

    def text_to_chat_completion(self) -> List[Dict[str, str]]:
        messages = []
        pattern = self._compile_regex(self.pattern)

        for match in pattern.finditer(self.prompt):
            groups = match.groupdict()
            message = self._process_message_group(groups)
            if message:
                messages.append(message)

        return messages

    def _process_message_group(self, groups: Dict[str, str]) -> Dict[str, str]:
        if groups.get("System"):
            return {"role": "system", "content": self._clean_system_content(groups["System"])}
        elif groups.get("User"):
            return {"role": "user", "content": self._clean_content(groups["User"])}
        elif groups.get("Assistant"):
            return {"role": "assistant", "content": self._clean_content(groups["Assistant"])}
        return {}

    @staticmethod
    def _clean_system_content(content: str) -> str:
        return content.replace("[INST]", "").replace("[/INST] Understood.</s>", "").strip()

    @staticmethod
    def _clean_content(content: str) -> str:
        return content.replace("\\", "").strip()

    def prepare_chat_completion(self) -> Tuple[List[Dict[str, str]], str]:
        chat = self.text_to_chat_completion()
        last_message = chat[-1]["content"]

        return chat, last_message

    def complete_chat_completion(self) -> str:
        spacing = "[/INST] " if self.prompt[self.last_index - 1 : self.last_index] != ">" else ""
        return f"{self.prompt[:self.last_index]}{spacing}{self.response}</s>[/INST]{self.character_raw}"
