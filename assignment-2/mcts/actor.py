from ..base import Actor


class MCTSActor(Actor):
    def get_action(self, state):
        ...

    def update(self):
        ...