from dataclasses import dataclass
from enum import Enum
from typing import Callable


@dataclass
class Routine:
    name: str
    routine: Callable


def _command_1(*args):
    print("command_1")
    return "command_1"


command_1 = Routine("1", routine=_command_1)


def _command_2(*args):
    print("command_2")
    return "command_2"


command_2 = Routine("2", routine=_command_2)


def _command_3(*args):
    print("command_3")
    return "command_3"


command_3 = Routine("3", routine=_command_3)


class Operation(Enum):
    command_1 = command_1
    command_2 = command_2
    command_3 = command_3
