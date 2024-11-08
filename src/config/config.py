import dataclasses

import yaml2pyclass


class Config(yaml2pyclass.CodeGenerator):
    @dataclasses.dataclass
    class ConfigurationClass:
        @dataclasses.dataclass
        class GeneralClass:
            @dataclasses.dataclass
            class StExtensionClass:
                enabled: bool
                text_to_chat_pattern: str
            
            dev_api_key: str
            write_thought_logs: bool
            write_full_logs: bool
            logs_path: str
            default_username: str
            prompt: str
            st_extension: StExtensionClass
        
        @dataclasses.dataclass
        class RegexClass:
            text_to_chat_pattern: str
            username_pattern: str
        
        @dataclasses.dataclass
        class HttpClientClass:
            timeout: int
        
        @dataclasses.dataclass
        class InferenceClass:
            @dataclasses.dataclass
            class TabbyApiClass:
                url: str
            
            @dataclasses.dataclass
            class MistralClass:
                api_key: str
                model: str
            
            @dataclasses.dataclass
            class OpenrouterClass:
                api_key: str
                model: str
            
            primary_url: str
            secondary_api_handler: str
            tabby_api: TabbyApiClass
            mistral: MistralClass
            openrouter: OpenrouterClass
        
        general: GeneralClass
        regex: RegexClass
        http_client: HttpClientClass
        inference: InferenceClass
    
    configuration: ConfigurationClass
