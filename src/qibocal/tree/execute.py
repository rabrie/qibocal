from dataclasses import dataclass

from .graph import Graph
from .history import History


@dataclass
class Executor:
    graph: Graph
    history: History

    @classmethod
    def load(cls, card):
        return cls(
            graph=Graph.load(card["actions"]),
            history=History([]),
        )

    def run(self):
        return
