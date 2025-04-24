import random
from abc import abstractmethod
from enum import Enum
from typing import Generator

import numpy as np
import torch
from torch import LongTensor, Tensor

from .utils import batched_bincount, sample_words


class Datasets(Enum):
    MASTERMIND = "mastermind"
    ARC = "arc"
    PCFG = "pcfg"
    SHIFT_CIPHER = "shift_cipher"

    @classmethod
    def list(cls):
        return [d.value for d in cls._member_map_.values()]


class Dataset:
    def __init__(self, n_tasks: int, n_samples: int, dataset_type: Datasets):
        self.n_tasks = n_tasks
        self.n_samples = n_samples
        self.dataset_type = dataset_type

    @abstractmethod
    def sample(self, seed: int | None = None):
        pass


class Mastermind(Dataset):
    def __init__(self, n_tasks: int, n_samples: int, code_length: int = 4, num_colours: int = 6):
        self.code_length = code_length
        self.num_colours = num_colours
        super().__init__(n_tasks, n_samples, Datasets.MASTERMIND)

    def sample(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        torch.manual_seed(seed)
        x = torch.randint(0, self.num_colours, (self.n_tasks, self.n_samples, self.code_length))
        code = torch.randint(0, self.num_colours, (self.n_tasks, self.code_length))
        full_correct = (code.unsqueeze(1) == x).sum(dim=-1)
        correct_colors = torch.min(
            batched_bincount(code, max_val=self.num_colours).unsqueeze(1),
            batched_bincount(x, max_val=self.num_colours),
        ).sum(dim=-1)
        y = torch.stack([full_correct, correct_colors], dim=-1)
        return x.numpy(), y.numpy()


class Arc(Dataset):
    def __init__(self, n_tasks: int, n_samples: int, n_vars: int = 10, n_vals: int = 10):
        self.n_vars = n_vars
        self.n_vals = n_vals
        super().__init__(n_tasks, n_samples, Datasets.ARC)

    def sample(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        torch.manual_seed(seed)
        pass


class PCFG(Dataset):
    def __init__(self, n_tasks: int, n_samples: int, n_vars: int = 10, n_vals: int = 10):
        self.n_vars = n_vars
        self.n_vals = n_vals
        super().__init__(n_tasks, n_samples, Datasets.PCFG)

    def sample(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        torch.manual_seed(seed)
        pass


class ShiftCipher(Dataset):
    def __init__(self, n_tasks: int, n_samples: int, encode: bool = True, corpus_name: str = "wordnet"):
        self.encode = encode
        self.corpus_name = corpus_name
        super().__init__(n_tasks, n_samples, Datasets.SHIFT_CIPHER)

    def sample(self, seed: int = 0) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        torch.manual_seed(seed)
        self.shift = torch.randint(0, 26, size=(self.n_tasks,))
        # self.start = torch.randint(0, 10, self.n_tasks)
        self.start = torch.zeros(self.n_tasks)
        # self.step = torch.randint(1, 10, self.n_tasks)
        self.step = torch.ones(self.n_tasks)
        random.seed(seed)
        word_sampler = sample_words(self.n_tasks, self.n_samples, self.corpus_name)
        for i in range(self.n_tasks):
            x = []
            y = []
            sampled_words = next(word_sampler)
            for _, word in enumerate(sampled_words):
                encoded_word = self.__shift_chipher(word, self.shift[i], self.start[i], self.step[i])
                if self.encode:
                    x.append(word)
                    y.append(encoded_word)
                else:
                    x.append(encoded_word)
                    y.append(word)
            yield x, y
        return

    def __shift_chipher(self, word: str, shift: int = 0, start: int = 0, step: int = 1) -> str:
        """
        Shift the letters in the word by the given shift value, starting from the given index and stepping by the given step value.
        """

        def shift_char(c, shift):
            if "A" <= c <= "Z":
                return chr((ord(c) - ord("A") + shift) % 26 + ord("A"))
            elif "a" <= c <= "z":
                return chr((ord(c) - ord("a") + shift) % 26 + ord("a"))
            return c  # Non-letter characters unchanged

        result = []
        for idx, c in enumerate(word):
            if idx >= start and (idx - start) % step == 0 and c.isalpha():
                c = shift_char(c, shift)
            result.append(c)
        return "".join(result)


DatasetMap = {
    Datasets.MASTERMIND: Mastermind,
    Datasets.ARC: Arc,
    Datasets.PCFG: PCFG,
    Datasets.SHIFT_CIPHER: ShiftCipher,
}


def get_dataset(dataset_type: str, n_tasks, n_samples, **kwargs) -> Dataset:
    dataset_type = Datasets._value2member_map_[dataset_type]
    return DatasetMap[dataset_type](n_tasks, n_samples, **kwargs)
