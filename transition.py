class Transition:
    def __init__(self, state, action, state_succ):
        self.s = state
        self.a = action
        self.s_succ = state_succ
        self.prob = 0
        self.reward = 0

    def get_state(self):
        return self.s

    def get_action(self):
        return self.a

    def get_succ_state(self):
        return self.s_succ

    def set_prob(self, prob):
        self.prob = prob

    def set_reward(self, reward):
        self.reward = reward

    def get_prob(self):
        return self.prob

    def get_reward(self):
        return self.reward
