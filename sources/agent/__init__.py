class AgentBase:
    def begin_episode(self):
        pass

    def act(self, observation: list[list[int]]):
        pass

    def end_episode(self):
        pass


from .dqn import DeepQNetworkAgent
