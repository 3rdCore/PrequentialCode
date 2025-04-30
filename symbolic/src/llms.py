import re
from abc import ABC, abstractmethod
from enum import Enum

import torch
from constants import MODEL_CONFIG_MASTERMIND
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from vllm import LLM, SamplingParams

from utils import get_available_gpus


class Model(Enum):
    GPT_35 = "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
    GPT_4 = "gpt-4", "gpt-4"
    GPT_4O = "gpt-4o", "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo", "gpt-4-turbo-preview"
    LLAMA_2 = "llama-2-7b", "meta-llama/Llama-2-7b-chat-hf"
    LLAMA_3 = "llama-3.1-8b", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MISTRAL_1 = "mistral-7b-v1", "mistralai/Mistral-7B-Instruct-v0.1"
    MISTRAL_2 = "mistral-7b-v2", "mistralai/Mistral-7B-Instruct-v0.2"
    MISTRAL_3 = "mistral-7b-v3", "mistralai/Mistral-7B-Instruct-v0.3"
    STARCHAT = "starchat-beta", "HuggingFaceH4/starchat-beta"

    def __new__(cls, *values):
        obj = object.__new__(cls)
        # first value is canonical value
        obj._value_ = values[0]
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values
        return obj

    def __repr__(self):
        return "<%s.%s: %s>" % (
            self.__class__.__name__,
            self._name_,
            ", ".join([repr(v) for v in self._all_values]),
        )

    @property
    def values(self):
        return self._all_values

    @classmethod
    def list(cls):
        return [m.value for m in cls._member_map_.values()]


class BaseLLM(ABC):
    def __init__(self, model_name: str, model_id):
        self.model_name = model_name
        self.model_id = model_id

    @abstractmethod
    def __call__(self, prompts, parser):
        pass

    @abstractmethod
    def gen_prompt(self, prompt):
        pass

    @abstractmethod
    def get_logprobs(self, response, value: str):
        pass

    @property
    def cls(self):
        return type(self).__name__


class GPT(BaseLLM):
    def __init__(self, model: Model = Model.GPT_4, **model_kwargs):
        super().__init__(model.values[0], model.values[1])
        self.llm = ChatOpenAI(model_name=self.model_id, temperature=0, n=1, top_logprobs=20, logprobs=True)

    def __call__(self, prompt, messages=[]):
        query = prompt["query"]
        if len(messages) == 0:
            system_message = prompt["system"]
            context_message = prompt["context"]
            messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=context_message))
        else:
            response = prompt["response"]
            messages.append(AIMessage(response))
            retry_message = prompt["retry_message"]
            query = retry_message
        messages.append(HumanMessage(content=query))
        reponse = self.llm.invoke(messages)

        return self.llm.invoke(messages), messages

    def gen_prompt(self, system, context, query):
        prompt = {"system": system, "context": context, "query": query}
        return prompt

    def extract_answer(self, response):
        return response.content.lower()

    def get_logprobs(self, response, values: list[str]):
        tokens = response.response_metadata["logprobs"]["content"]
        logprobs_arr = []
        for value in values:
            token_logprobs = tokens[[token["token"] for token in tokens].index(value)]
            logprobs = dict(map(lambda x: (x["token"], x["logprob"]), token_logprobs["top_logprobs"]))
            logprobs_arr.append(logprobs)
        return logprobs_arr


##### vLLMS #####
class VLLM(BaseLLM):
    def __init__(self, model: Model = Model.LLAMA_3, **model_kwargs):
        super().__init__(model.values[0], model.values[1])
        num_gpus = get_available_gpus()
        self.llm = LLM(model=self.model_id, tensor_parallel_size=max(1, num_gpus))
        self.sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1, logprobs=20)

    def __call__(self, prompts, messages=[]):
        responses = self.llm.generate(prompts, self.sampling_params)
        return responses, messages

    def gen_prompt(self, system, context, query):
        prompt = f"{system}{context}{query}"
        return prompt

    def extract_answer(self, response):
        return response.outputs[0].text.strip()

    def get_logprobs(self, response, values):
        logprobs = [
            [
                (logprob.decoded_token, round(logprob.logprob, 4))
                for (token, logprob) in response.outputs[0].logprobs[0].items()
            ]
            for value in values
        ]
        return logprobs


# use dictionary multimap.
def get_model(model_name: str, model_config):
    model: Model = Model._value2member_map_[model_name]
    llm = GPT if model in [Model.GPT_35, Model.GPT_4, Model.GPT_4O, Model.GPT_4_TURBO] else VLLM
    return llm(model, **model_config)
