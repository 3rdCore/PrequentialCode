import re
from abc import abstractmethod

import numpy as np
from dataset import Datasets
from templates import MastermindTemplate

from utils import compute_random_baseline, cross_entropy_loss, softmax


class Parser:
    def __init__(self, dataset_type: Datasets):
        self.dataset_type = dataset_type

    @abstractmethod
    def parse(self, response, options):
        pass

    @abstractmethod
    def process(self, answer, logprobs):
        pass


class MastermindParser(Parser):
    def __init__(self, dataset_type: Datasets):
        super().__init__(Datasets.MASTERMIND)
        self.template = MastermindTemplate()

    def parse(self, response) -> tuple[str, bool, str]:
        try:
            response = response.content.lower()
            answers = self.extract_answer(response)
            if len(answers) != 2:
                return None, False, self.template.ERROR_MESSAGE
            return answers, True, None
        except Exception as err:
            return response, False, self.template.ERROR_MESSAGE

    def extract_answer(self, response):
        matches = re.findall(self.template.PATTERN, response)
        return matches

    def process(self, answer, logprobs):
        answer = [int(a) for a in answer]
        logprobs_arr = []
        for logprob in logprobs:
            d_value = min(logprob.values())
            logprobs_arr.append([logprob.get(v, d_value) for v in self.template.VALUES])
        return answer, logprobs_arr


class ArcParser(Parser):
    def __init__(self, dataset_type: Datasets):
        super().__init__(Datasets.ARC)

    def parse(self, response):
        pass


class PCFGParser(Parser):
    def __init__(self, dataset_type: Datasets):
        super().__init__(Datasets.PCFG)

    def parse(self, response):
        pass


ParserMap = {Datasets.MASTERMIND: MastermindParser, Datasets.ARC: ArcParser, Datasets.PCFG: PCFGParser}


def get_parser(dataset_type: str) -> Parser:
    dataset_type = Datasets._value2member_map_[dataset_type]
    parser = ParserMap[dataset_type]
    return parser(dataset_type)


def process_results(results, data):
    y_true = data[1]
    y_pred = []
    logprobs = []
    for result in results:
        y_pred.append(list(map(lambda r: r["answer"], result)))
        logprobs.append(list(map(lambda r: [softmax(l) for l in r["logprobs"]], result)))
    y_pred = np.array(y_pred)
    logprobs = np.array(logprobs)
    y_true = y_true[:, 1:, :]
    loss = cross_entropy_loss(y_true[..., 0], logprobs[..., 0, :], keepdims=True) + cross_entropy_loss(
        y_true[..., 1], logprobs[..., 1, :]
    )
    probs = np.exp(-loss)
    loss = loss / 2
    probs_rand = compute_random_baseline(y_true)

    acc = (y_pred == y_true).sum(axis=-1) == 2
    task_wise_acc = acc.mean(axis=1)
    n_sample_loss = loss.mean(axis=0)
    metrics = {
        "n_sample_loss_nexttoken": n_sample_loss.tolist(),
        "probs": probs.mean(axis=0).tolist(),
        "probs_rand": probs_rand.mean(axis=0).tolist(),
        "cross_entropy_loss": n_sample_loss.mean(),
        "task_wise_accuracy": task_wise_acc.tolist(),
        "accuracy": acc.mean(),
    }
    return metrics, y_pred, logprobs, probs
