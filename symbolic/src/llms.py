import re
from abc import ABC, abstractmethod
from enum import Enum
from warnings import warn

import numpy as np
from constants import LOGIT_BIAS, OPTIONS
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from utils import softmax


class Model(Enum):
    GPT_35 = "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
    GPT_4 = "gpt-4", "gpt-4"
    GPT_4O = "gpt-4o", "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo", "gpt-4-turbo-preview"
    LLAMA = "llama-2-7b", "meta-llama/Llama-2-7b-chat-hf"
    LLAMA_3 = "llama-3.1-8b", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MISTRAL = "mistral-7b", "mistralai/Mixtral-8x7B-Instruct-v0.1"
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


def extract_html_tags(text, keys):
    content_dict = {}
    keys = set(keys)
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        # pattern = r"\b(?:[0-4])\b"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


def find_answer(text):
    pattern = r"\b(?:[0-4])\b"
    # pattern = r"\b(?:[a-e])\b"

    matches = re.findall(pattern, text)
    return matches


def get_logprobs(response, value):
    tokens = response.response_metadata["logprobs"]["content"]
    token_logprobs = tokens[[token["token"] for token in tokens].index(value)]
    logprobs = dict(map(lambda x: (x["token"], x["logprob"]), token_logprobs["top_logprobs"]))
    # print(response.content, logprobs)
    return logprobs


def parse_response(response, parse_failed=False):
    try:
        input_string = response.content.lower()
        answer = extract_html_tags(input_string.lower(), ["answer"])["answer"][0]
        if answer not in OPTIONS:
            return (
                answer,
                False,
                "Error: An <answer></answer> tag was found, but the contained value is invalid. The only accepted values are options 0-4.",
            )
    except Exception as err:
        matches = find_answer(input_string.lower())
        if parse_failed and matches:
            return (matches[0], True, None)

        return (
            input_string,
            False,
            f"Error: Failed to extract the answer - missing <answer></answer> tag. Please provide the answer in <answer></answer> tag. The only accepted values are options 0-4.",
        )
    return (answer, True, None)


class BaseLLM(ABC):
    def __init__(self, model_name: str, model_id):
        self.model_name = model_name
        self.model_id = model_id

    def __call__(self, prompts):
        return self._query2(prompts)

    @abstractmethod
    def _query(self, prompts):
        pass

    @property
    def cls(self):
        return type(self).__name__


class GPT(BaseLLM):
    def __init__(self, model: Model = Model.GPT_4, temperature=1, n=1, top_logprobs=5, max_tokens=20):
        super().__init__(model.values[0], model.values[1])
        self.llm = ChatOpenAI(
            model_name=self.model_id,
            temperature=temperature,
            n=n,
            logit_bias=LOGIT_BIAS,
            logprobs=True,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )

    def _query(self, prompts, n_retry=10):
        results = []
        success, success_retry, failure = 0, 0, 0
        system_message = prompts["system"]
        context_message = ""
        context = prompts["context"]
        queries = prompts["query"]
        success_no_retry, success_retry, failure = 0, 0, 0
        for i, query in tqdm(enumerate(queries), total=len(queries), desc="Prompts: "):
            messages = []
            context_message += context[i]
            messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=context_message))
            messages.append(HumanMessage(content=query))
            print(len(system_message), len(context_message), len(query))
            for j in range(n_retry + 1):
                response = self.llm(messages)
                value, valid, retry_message = parse_response(response, j != 0)
                if valid:
                    success_no_retry += int(j == 0)
                    success_retry += int(j != 0)
                    logprobs = get_logprobs(response, value)
                    # if not all_tokens_present:
                    #     msg = f"Error - query {i} failed. Not all tokens present in logprobs."
                    #     warn(msg, RuntimeWarning)
                    #     continue
                    break
                msg = f"Query {i} failed. Retrying {j+1}/{n_retry}.\n[LLM]:\n{response.content}\n[User]:\n{retry_message}"
                warn(msg, RuntimeWarning)

            if not valid:
                failure += 1
                value = "err"
                msg = f"Error - query {i} failed. Could not parse response after {n_retry} retries."
                warn(msg, RuntimeWarning)
                logprobs = None
            results.append({"answer": value, "logprobs": logprobs, "response": response})
        query_log = (success_no_retry, success_retry, failure)
        return results, query_log

    def _query2(self, prompts, n_retry=10):
        results = []
        success, success_retry, failure = 0, 0, 0
        system_message = prompts["system"]
        context_message = ""
        contexts = prompts["context"]
        queries = prompts["query"]
        success_no_retry, success_retry, failure = 0, 0, 0
        for i, context in tqdm(enumerate(contexts), total=len(contexts), desc="Prompts: "):
            context_message += context
            query_results = []
            with open(f"../experiments/results/logs/context_{i}.txt", "w") as f:
                f.write(context_message)

            for j, query in enumerate(queries):
                messages = []
                messages.append(SystemMessage(content=system_message))
                messages.append(HumanMessage(content=context_message))
                messages.append(HumanMessage(content=query))
                for j in range(n_retry + 1):
                    response = self.llm(messages)
                    value, valid, retry_message = parse_response(response, j != 0)
                    if valid:
                        success_no_retry += int(j == 0)
                        success_retry += int(j != 0)
                        logprobs = get_logprobs(response, value)
                        # if not all_tokens_present:
                        #     msg = f"Error - query {i} failed. Not all tokens present in logprobs."
                        #     warn(msg, RuntimeWarning)
                        #     continue
                        break
                    msg = f"Query {i} failed. Retrying {j+1}/{n_retry}.\n[LLM]:\n{response.content}\n[User]:\n{retry_message}"
                    warn(msg, RuntimeWarning)

                if not valid:
                    failure += 1
                    value = "err"
                    msg = f"Error - query {i} failed. Could not parse response after {n_retry} retries."
                    warn(msg, RuntimeWarning)
                    logprobs = None
                query_results.append({"answer": value, "logprobs": logprobs, "response": response})
            results.append(query_results)
        query_log = (success_no_retry, success_retry, failure)
        return results, query_log
