import re
from abc import abstractmethod

import numpy as np
from dataset import Datasets
from templates import MastermindTemplate, ShiftCipherTemplate

from utils import compute_marginal_baseline, compute_random_baseline, cross_entropy_loss, softmax


class Parser:
    def __init__(self, dataset_type: Datasets, extract_answer):
        self.dataset_type = dataset_type
        self.get_response_text = extract_answer

    @abstractmethod
    def parse(self, response, options):
        pass

    def extract_answer(self, response_text):
        matches = re.findall(self.template.PATTERN, response_text)
        return matches

    @abstractmethod
    def process(self, answer, logprobs):
        pass


class MastermindParser(Parser):
    def __init__(self, dataset_type: Datasets, extract_answer=lambda x: x):
        self.extract = extract_answer
        super().__init__(Datasets.MASTERMIND, extract_answer)
        self.template = MastermindTemplate()

    def parse(self, response) -> tuple[str, bool, str]:
        try:
            response_text = self.get_response_text(response)
            answers = self.extract_answer(response_text)
            if len(answers) != 2:
                return None, False, self.template.ERROR_MESSAGE
            return answers, True, None
        except Exception as err:
            return response, False, self.template.ERROR_MESSAGE

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


class ShiftCipherParser(Parser):
    def __init__(self, dataset_type: Datasets, extract_answer=lambda x: x):
        super().__init__(Datasets.SHIFT_CIPHER, extract_answer=extract_answer)
        self.template = ShiftCipherTemplate()

    def parse(self, response) -> tuple[str, bool, str]:
        try:
            response_text = self.get_response_text(response)
            answers = self.extract_answer(response_text)
            if len(answers) != 1:
                return response_text, False, self.template.ERROR_MESSAGE
        except Exception as err:
            print(f"Error parsing response - ({response_text}):", err)
            return response_text, False, self.template.ERROR_MESSAGE

        return answers[0], True, None

    def process(self, answer, logprobs):
        return answer, logprobs


ParserMap = {
    Datasets.MASTERMIND: MastermindParser,
    Datasets.ARC: ArcParser,
    Datasets.PCFG: PCFGParser,
    Datasets.SHIFT_CIPHER: ShiftCipherParser,
}


def get_parser(dataset_type: str, extract_answer) -> Parser:
    dataset_type = Datasets._value2member_map_[dataset_type]
    parser = ParserMap[dataset_type]
    return parser(dataset_type, extract_answer)


def process_results(results, data):
    y_trues = [d[1][1:] for d in data]
    y_preds = [list(map(lambda r: r["answer"][0], result)) for result in results]

    def norm_prob(r):
        r_token, r_logprobs = tuple(map(list, zip(*r["logprobs"][0])))
        r_logprobs = np.array(r_logprobs)
        r_logprobs -= np.log(np.sum(np.exp(r_logprobs)))
        return list(zip(r_token, r_logprobs.tolist()))

    logprob_arrs = [list(map(lambda r: norm_prob(r), result)) for result in results]
    logprob_of_y_trues = []
    for i, (y_true, logprobs) in enumerate(zip(y_trues, logprob_arrs)):
        logprobs_of_y_true = []

        for j, (value, logprob) in enumerate(zip(y_true, logprobs)):
            print(f"y_true: {y_true}, logprobs: {logprob}")
            logprob_dict = dict(logprob)
            logprob_of_value = -logprob_dict[value] if value in logprob_dict else -logprob[-1][1]
            logprobs_of_y_true.append(logprob_of_value)
        logprob_of_y_trues.append(logprobs_of_y_true)

    # print(f"y_true: {y_trues}")
    # print(f"y_pred: {y_preds}")
    # print(f"logprobs: {logprob_of_y_trues}")

    y_pred = np.array(y_preds)
    logprobs = np.array(logprob_of_y_trues)
    probs = np.exp(-logprobs).round(4)

    acc = y_pred == y_true
    task_wise_acc = acc.mean(axis=1).round(4)
    n_sample_acc = acc.mean(axis=0).round(4)
    n_sample_loss = logprobs.mean(axis=0).round(4)
    metrics = {
        "n_sample_loss_nexttoken": n_sample_loss.tolist(),
        "probs": probs.mean(axis=0).tolist(),
        "cross_entropy_loss": n_sample_loss.mean().round(4),
        "task_wise_accuracy": task_wise_acc.tolist(),
        "n_sample_accuracy": n_sample_acc.tolist(),
        "accuracy": acc.mean().round(4),
    }
    return metrics, y_pred, logprobs, probs


# def process_results(results, data):
#     code_length = data[0].shape[-1]
#     y_true = data[1]
#     y_pred = []
#     logprobs = []
#     for result in results:
#         y_pred.append(list(map(lambda r: r["answer"], result)))
#         logprobs.append(list(map(lambda r: [softmax(l) for l in r["logprobs"]], result)))
#     y_pred = np.array(y_pred)
#     logprobs = np.array(logprobs)
#     y_true = y_true[:, 1:, :]
#     loss = cross_entropy_loss(y_true[..., 0], logprobs[..., 0, :], keepdims=True) + cross_entropy_loss(
#         y_true[..., 1], logprobs[..., 1, :]
#     )
#     probs = np.exp(-loss)
#     loss = loss / 2
#     probs_rand = compute_random_baseline(y_true)
#     loss_margin, probs_margin = compute_marginal_baseline(y_true, code_length)

#     acc = (y_pred == y_true).sum(axis=-1) == 2
#     task_wise_acc = acc.mean(axis=1)
#     n_sample_loss = loss.mean(axis=0)
#     n_sample_loss_margin = loss_margin.mean(axis=0)
#     metrics = {
#         "n_sample_loss_nexttoken": n_sample_loss.tolist(),
#         "n_sample_loss_nexttoken_marg": n_sample_loss_margin.tolist(),
#         "probs": probs.mean(axis=0).tolist(),
#         "probs_marg": probs_margin.mean(axis=0).tolist(),
#         "probs_rand": probs_rand.mean(axis=0).tolist(),
#         "cross_entropy_loss": n_sample_loss.mean(),
#         "task_wise_accuracy": task_wise_acc.tolist(),
#         "accuracy": acc.mean(),
#     }
#     return metrics, y_pred, logprobs, probs
