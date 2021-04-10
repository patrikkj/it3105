from ..base import Actor


class MCTSActor(Actor):
    def __init__(self, *args, mode='network'):
        self.mode = mode

    def get_action(self, state):
        ...

    def update(self):
        ...
