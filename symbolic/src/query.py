from warnings import warn

import pandas as pd
from dataset import Datasets
from llms import get_model
from process import Parser, get_parser
from tqdm import tqdm


class Query:
    def __init__(self, model_name: str, dataset_type: str, n_retry=10):
        self.model_name = model_name
        self.n_retry = n_retry
        self.parser = get_parser(dataset_type)
        self.model = get_model(model_name)
        self.success_no_retry = 0
        self.success_retry = 0
        self.failure = 0

    def query_prompts(self, system: str, task_prompts: list[pd.DataFrame]):
        task_results = []
        for t, prompts in enumerate(task_prompts):
            results, query_stats = self.query(system, prompts)
            task_results.append(results)
        return task_results, query_stats

    def query(self, system: str, prompts: pd.DataFrame) -> list[str]:
        results = []
        context = prompts["context"]
        queries = prompts["query"]
        context_message = ""
        for i, query in tqdm(enumerate(queries[1:]), total=len(queries) - 1, desc="Prompts: "):
            context_message += context[i]
            prompt = {"system": system, "context": context_message, "query": query}
            messages = []
            for j in range(self.n_retry + 1):
                response, messsages = self.model(prompt, messages=messages)
                answer, valid, retry_message = self.parser.parse(response)
                self.update_query_stats(valid, attempt=j)
                if valid:
                    answer = [answer] if type(answer) == str else answer
                    answer, logprobs = self.parser.process(answer, self.model.get_logprobs(response, answer))
                    answer = {"answer": answer, "logprobs": logprobs}
                    break
                msg = f"Query {i} failed. Retrying {j+1}/{self.n_retry}.\n[LLM]:\n{response.content}\n[User]:\n{retry_message}"
                warn(msg, RuntimeWarning)
                prompt["response"] = response.content
                prompt["retry_message"] = retry_message

            if not valid:
                answer = "err"
                msg = f"Error - query {i} failed. Could not parse response after {self.n_retry} retries."
                answer = {"answer": answer, "logprobs": None}
                warn(msg, RuntimeWarning)
            results.append(answer)

        query_stats = self.get_query_stats()
        return results, query_stats

    def update_query_stats(self, valid, attempt: int):
        if valid:
            if attempt == 0:
                self.success_no_retry += 1
            else:
                self.success_retry += 1
        elif attempt == self.n_retry:
            self.failure += 1

    def get_query_stats(self):
        return self.success_no_retry, self.success_retry, self.failure
