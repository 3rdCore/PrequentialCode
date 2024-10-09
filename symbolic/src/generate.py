from abc import ABC, abstractmethod

import pandas as pd
from dataset import Dataset, Datasets
from templates import ArcTemplate, MastermindTemplate, PCFGTemplate


class PromptGenerator(ABC):
    def __init__(self, with_option=False) -> None:
        self.with_option = with_option

    @abstractmethod
    def generate(self, data) -> tuple[str, list[pd.DataFrame]]:
        pass

    @abstractmethod
    def generate_prompt(self, i: int, x, y) -> tuple[str, str]:
        pass


class MastermindGenerator(PromptGenerator):
    def __init__(self, with_option=False) -> None:
        super().__init__(with_option)
        self.template = MastermindTemplate(with_option)

    def generate(self, data) -> tuple[str, list[pd.DataFrame]]:
        system = self.template.SYSTEM
        task_prompts = []
        X, Y = data
        for t in range(len(X)):
            x_samples = X[t]
            y_samples = Y[t]
            contexts = []
            queries = []
            for s, (x, y) in enumerate(zip(x_samples, y_samples)):
                context, query = self.generate_prompt(s, x, y)
                contexts.append(context)
                queries.append(query)
            task_prompts.append(pd.DataFrame({"context": contexts, "query": queries}))
        return system, task_prompts

    def generate_prompt(self, i: int, x, y) -> tuple[str, str]:
        x = " ".join([str(xi) for xi in x])
        y = " ".join([str(yi) for yi in y])
        context = self.template.CONTEXT.format(input=x, output=y)
        prompt = self.template.QUERY.format(input=x, output=y)
        return context, prompt


class ArcGenerator(PromptGenerator):
    def __init__(self, with_option=False) -> None:
        super().__init__(with_option)
        self.template = ArcTemplate(with_option)

    def generate(self, data) -> tuple[str, list[pd.DataFrame]]:
        pass

    def generate_prompt(self, i: int, x, y) -> tuple[str, str]:
        pass


class PCFGGenerator(PromptGenerator):
    def __init__(self, with_option=False) -> None:
        super().__init__(with_option)
        self.template = ArcTemplate(with_option)

    def generate(self, data) -> tuple[str, list[pd.DataFrame]]:
        pass

    def generate_prompt(self, i: int, x, y) -> tuple[str, str]:
        pass


GeneratorMap = {
    Datasets.MASTERMIND: MastermindGenerator,
    Datasets.ARC: ArcGenerator,
    Datasets.PCFG: PCFGGenerator,
}


def generate_prompts(dataset_type: str, data, with_option=False) -> tuple[str, list[pd.DataFrame]]:
    prompt_generator = GeneratorMap[Datasets._value2member_map_[dataset_type]](with_option)
    return prompt_generator.generate(data)
