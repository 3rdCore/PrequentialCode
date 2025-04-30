import os
from string import Template

TEMPLATES_PATH = os.path.join("../data/templates")


class MastermindTemplate:
    def __init__(self, with_option=False) -> None:
        self.code_length = 8
        with open(os.path.join(TEMPLATES_PATH, f"mastermind_description_s{self.code_length}.md")) as f:
            self.SYSTEM = f.read()
        self.CONTEXT = "Guess: {input}\nResponse: {output}\n\n"
        self.RESPONSE_FORMAT = f"What do you think the response is for this final guess? Make sure to reply with just 2 digits between 0-{self.code_length}, separated by a single space character. Just provide the 2 digits"
        self.QUERY = "Guess: {input}\nResponse: ? ?\n-----------\n\n" + self.RESPONSE_FORMAT
        self.ERROR_MESSAGE = f"Answer not in the expected format.\n Make sure to reply with just 2 digits between 0-{self.code_length}, separated by a single space character. Just provide the 2 digits"
        self.VALUES = list(map(str, range(self.code_length + 1)))
        self.PATTERN = r"\b(?:[0-8])\b"


class ArcTemplate:
    def __init__(self, with_option=False) -> None:
        pass


class PCFGTemplate:
    def __init__(self, with_option=False) -> None:
        pass


class ShiftCipherTemplate:
    def __init__(self, is_encode: bool = True, with_option=False) -> None:
        self.is_encode = is_encode
        with open(os.path.abspath(os.path.join(TEMPLATES_PATH, f"shift_cipher_description.md"))) as f:
            self.SYSTEM = f.read()
        self.CONTEXT = "X:{input}\nY:{output}\n\n"
        self.QUERY = "X:{input}\nY:"
        # self.CONTEXT = "X:{input} -> Y:{output}\n"
        # self.QUERY = "X:{input} -> Y: ??? (you must predict Y)"

        self.ERROR_MESSAGE = f"Answer not in the expected format.\n Make sure to reply with just the Y."
        self.VALUES = list(map(str, range(1, 26)))
        self.PATTERN = r"\b([a-zA-Z]+)\b"
