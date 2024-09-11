import os

TEMPLATES_PATH = os.path.join("../data/templates")


class MastermindTemplate:
    def __init__(self, with_option=False) -> None:
        with open(os.path.join(TEMPLATES_PATH, "mastermind_description.md")) as f:
            self.SYSTEM = f.read()
        self.CONTEXT = "{input}\n{output}\n----\n"
        self.RESPONSE_FORMAT = "Just provide 2 values separated by a space: count of correct colors and count of exact matches. The only accepted values are numbers from 0-4."
        self.QUERY = self.RESPONSE_FORMAT + "\n{input}\n"
        self.ERROR_MESSAGE = "Answer not in the expected format. "
        self.VALUES = list(map(str, range(5)))
        self.PATTERN = r"\b(?:[0-4])\b"


class ArcTemplate:
    def __init__(self, with_option=False) -> None:
        pass


class PCFGTemplate:
    def __init__(self, with_option=False) -> None:
        pass
