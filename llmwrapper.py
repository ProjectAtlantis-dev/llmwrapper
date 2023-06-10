

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from pydantic import Extra, Field, root_validator

import requests
import json


from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage
)








# a langchain LLM just handles text - you probaby want a chatmodel instead
class CustomLLM(LLM):

    service_name: str
    model_name: str

    @property
    def _llm_type(self) -> str:
        return self.service_name + "." + self.model_name

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
                "service": self.service_name,
                "model":  self.model_name,
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

        if "error" in response_dict:
            raise ValueError(response_dict["error"])

        # langchain requires a string response
        return response_dict["completion"]


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}



def testLLM():

    llm = CustomLLM(service_name="openai", model_name="gpt-4")
    result = llm(
    '''In less than 10 words, answer the following questions:
    Do you know the way to San Jose?''')
    print(result)

    #result = llm("You can really breathe in San Jose")
    #print(result)

    #result = llm("I got lots of friends in San Jose")
    #print(result)



class CustomChatModel(BaseChatModel):
    service_name: str
    model_name: str

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return self.service_name + "-" + self.model_name

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        llm = CustomLLM(service_name=self.service_name, model_name=self.model_name)
        prompt = "".join(message.content for message in messages)
        content = llm(prompt)
        return ChatResult(generations=[ChatGeneration(message=SystemMessage(content=content))])


    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=SystemMessage(content="not used"))])




def testChatModel():
    #chat_model = CustomChatModel(service_name="openai", model_name="text-davinci-002-render-sha")
    chat_model = CustomChatModel(service_name="poe", model_name="Claude-instant")
    message = HumanMessage(content="describe Sandestin FL in just 5 words or less")
    print(chat_model([message]).content)






if __name__ == "__main__":
    #testLLM()
    testChatModel()

