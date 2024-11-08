from src import Config, InferenceBase


class TabbyApiInference(InferenceBase):
    def __init__(self, config: Config):
        super().__init__(config)

    def completion(self, completion_request: dict, stored_last_message: str):
        pass
