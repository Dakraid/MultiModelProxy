from httpx import Timeout

from src import Config


class Variables:
    def __init__(self, config: Config, timeout: Timeout):
        self.primary_url = config.configuration.inference.primary_url
        self.timeout = timeout
        self.last_message = ""
        self.last_expanded = ""
