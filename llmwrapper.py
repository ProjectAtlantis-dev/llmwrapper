

import requests
import json

from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM



class CustomLLM(LLM):

    service: str
    model: str

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        response = requests.post(
            url="http://localhost:3010/llm",
            json={
                "service": self.service,
                "model":  self.model,
                "prompt": prompt
            },
        )

        #print(response.content)

        if response.status_code != 200:
            optional_detail = response.text
            raise ValueError(
                f"LLM call failed with status code {response.status_code}. "
                f"Details: {optional_detail}"
            )

        #response_text = response.text
        #return response_text
        response_dict = response.json()

        if response_dict["error"]:
            raise ValueError(response_dict["error"])

        # langchain requires a string response
        return response_dict["completion"]


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}



llm = CustomLLM(service="poe", model="GPT-4")

result = llm("Do you know the way to San Jose?")

print(result)

result = llm("You can really breathe in San Jase")

print(result)

result = llm("I got lots of friends in San Jose")

print(result)
