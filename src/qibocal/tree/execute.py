from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Union

from .graph import Graph
from .history import Completed, History
from .runcard import Id, Runcard
from .task import Task


@dataclass
class Executor:
    graph: Graph
    history: History
    head: Optional[Id] = None
    pending: Set[Id] = field(default_factory=set)

    @classmethod
    def load(cls, card: Union[dict, Path]):
        runcard = Runcard.load(card)

        return cls(graph=Graph.from_actions(runcard.actions), history=History({}))

    def available(self, task: Task):
        for pred in self.graph.predecessors(task.id):
            ptask = self.graph.task(pred)

            if ptask.uid not in self.history:
                return False

        return True

    def successors(self, task: Task):
        succs: List[Task] = []

        if task.main is not None:
            # main task has always more priority on its own, with respect to
            # same with the same level
            succs.append(self.graph.task(task.main))
        # add all possible successors to the list of successors
        succs.extend([self.graph.task(id) for id in task.next])

        return succs

    def next(self) -> Optional[Id]:
        candidates = self.successors(self.current)

        if len(candidates) == 0:
            candidates.extend([])

        candidates = list(filter(lambda t: self.available(t), candidates))

        # sort accord to priority
        candidates.sort(key=lambda t: t.priority)
        if len(candidates) != 0:
            self.pending.update([t.id for t in candidates[1:]])
            return candidates[0].id

        availables = list(
            filter(lambda t: self.available(self.graph.task(t)), self.pending)
        )
        if len(availables) == 0:
            if len(self.pending) == 0:
                return None
            raise RuntimeError("")

        selected = min(availables, key=lambda t: self.graph.task(t).priority)
        self.pending.remove(selected)
        return selected

    @property
    def current(self):
        assert self.head is not None
        return self.graph.task(self.head)

    def run(self):
        self.head = self.graph.start

        while self.head is not None:
            task = self.current

            output = task.run()
            completed = Completed(task, output)
            self.history.push(completed)

            self.head = self.next()
