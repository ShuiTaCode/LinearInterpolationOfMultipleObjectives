import numpy as np
from transition import Transition
from state import State
from numpy import random


def delta_x(state1, state2):
    return np.abs(state1.get_x() - state2.get_x())


def delta_y(state1, state2):
    return np.abs(state1.get_y() - state2.get_y())


def compare_states_from_different_mdp(state1, state2):
    print('compared states', state1, state2)
    return state1.get_x() == state2.get_x() and state1.get_y() == state2.get_y()


class Mdp:
    def create_set_of_states(self):
        result = []
        for i in range(self.size):
            for j in range(self.size):
                result.append(State(i, j))
        return result

    def create_set_of_transitions(self):
        result = []
        for state in self.init_set_of_states:
            for action in self.init_set_of_actions:
                for succ_state in self.init_set_of_states:
                    t = Transition(state, action, succ_state)
                    if delta_x(t.get_state(), t.get_succ_state()) + delta_y(t.get_state(),
                                                                            t.get_succ_state()) < 2:
                        result.append(Transition(state, action, succ_state))

        return result

    def calculate_prob(self, transition):
        left_border = transition.s.x == 0
        right_border = transition.s.x == self.size - 1
        top_border = transition.s.y == 0
        bottom_border = transition.s.y == self.size - 1

        if transition.a == 'exit':
            if transition.s == self.end_state:
                return 1
            else:
                return 0

        if transition.a == self.init_set_of_actions[0]:  # action = up
            if transition.s_succ == transition.s:  # stay
                p = 0.05
                if top_border:
                    p += 0.80
                if bottom_border:
                    p += 0.05
                if left_border or right_border:
                    p += 0.05
                return p
            if transition.s_succ.x == transition.s.x + 1:  # moving right
                return 0.05
            if transition.s_succ.x == transition.s.x - 1:  # moving left
                return 0.05
            if transition.s_succ.y == transition.s.y - 1:  # moving up
                return 0.80
            if transition.s_succ.y == transition.s.y + 1:  # moving down
                return 0.05

        if transition.a == self.init_set_of_actions[1]:  # down
            if transition.s_succ == transition.s:
                p = 0.05
                if bottom_border:
                    p += 0.80
                if top_border:
                    p += 0.05
                if left_border or right_border:
                    p += 0.05
                return p
            if transition.s_succ.x == transition.s.x + 1:  # moving right
                return 0.05
            if transition.s_succ.x == transition.s.x - 1:  # moving left
                return 0.05
            if transition.s_succ.y == transition.s.y - 1:  # moving up
                return 0.05
            if transition.s_succ.y == transition.s.y + 1:  # moving down
                return 0.80

        if transition.a == self.init_set_of_actions[2]:  # left
            if transition.s_succ == transition.s:
                p = 0.05
                if left_border:
                    p += 0.80
                if right_border:
                    p += 0.05
                if top_border or bottom_border:
                    p += 0.05
                return p
            if transition.s_succ.x == transition.s.x + 1:  # moving right
                return 0.05
            if transition.s_succ.x == transition.s.x - 1:  # moving left
                return 0.80
            if transition.s_succ.y == transition.s.y - 1:  # moving up
                return 0.05
            if transition.s_succ.y == transition.s.y + 1:  # moving down
                return 0.05

        if transition.a == self.init_set_of_actions[3]:  # right
            if transition.s_succ == transition.s:
                p = 0.05
                if right_border:
                    p += 0.80
                if left_border:
                    p += 0.05
                if top_border or bottom_border:
                    p += 0.05
                return p
            if transition.s_succ.x == transition.s.x + 1:  # moving right
                return 0.80
            if transition.s_succ.x == transition.s.x - 1:  # moving left
                return 0.05
            if transition.s_succ.y == transition.s.y - 1:  # moving up
                return 0.05
            if transition.s_succ.y == transition.s.y + 1:  # moving down
                return 0.05

    def create_transition_probability(self):
        result = []
        for t in self.init_set_of_transitions:
            t.set_prob(round(self.calculate_prob(t), 2))
            result.append(t)

        return result

    def create_transition_reward(self):
        result = []
        print('länge', len(self.init_set_of_transitions_probabilities))
        for t in self.init_set_of_transitions_probabilities:
            if t.get_succ_state() == self.end_state:
                t.set_reward(1)
            print('probs', t.get_state().__dict__, t.get_action(), t.get_succ_state().__dict__, t.get_prob(),
                  t.get_reward())
            result.append(t)

        return result

    def solve_mdp(self):
        while len([s for s in self.init_set_of_states if (s.get_value()['r'] > 0)]) < self.size * self.size - 1:
            self.run_iteration()
        arr = []
        return self.init_set_of_states

    def return_max_val(self, arr):
        output = arr[0]
        for obj in arr:
            if obj['r'] > output['r']:
                output = obj
        return output

    def get_value_by_policy(self, arr):
        print('das ist das array')
        for t in arr:
            print(t['s'].__dict__, t)
        state_from_arr = arr[0]['s']  # arr hat immer gleichen startpunkt s
        action = ""
        if arr[0]['a'] == 'exit':
            return arr[0]
        for state in self.policy:
            if compare_states_from_different_mdp(state, state_from_arr):
                print('gefunden', state.get_value()['a'])
                action = state.get_value()['a']

        if action == 'exit':  # wenn policy im entzustand nehme einfach erste lösung
            return arr[0]

        print('das sind nun all tr einer action',[obj for obj in arr if
                                    (obj['a'] == action)])
        return self.return_max_val([obj for obj in arr if
                                    (obj['a'] == action)])

        # print('es ist ein fehler aufgetreten das wurde nicht gefunden: ',action)

    def run_transformation(self):
        print('transformation wird ausgeführt')
        result = []
        for state in self.init_set_of_states:
            arr = []
            for tr in [t for t in self.init_set_of_transitions_probabilities_and_rewards if
                       (t.get_state() == state)]:
                arr.append({
                    's': tr.get_state(),
                    'a': tr.get_action(),
                    'r': tr.get_prob() * (tr.get_reward() + self.gamma * tr.get_succ_state().get_value()['r'])
                })

            if len(self.policy) > 0:
                result.append(self.get_value_by_policy(arr))
                # print('optimal action', self.get_value_by_policy(arr))
        print('result length', len(result))
        for x in range(len(self.init_set_of_states)):
            # print('jojojo',result[x]['s'].__dict__,result[x])
            self.init_set_of_states[x].set_value(result[x])

    def run_iteration(self):
        print('iteration wird ausgeführt')
        result = []
        for state in self.init_set_of_states:
            arr = []
            for tr in [t for t in self.init_set_of_transitions_probabilities_and_rewards if
                       (t.get_state() == state)]:
                arr.append({
                    's': tr.get_state(),
                    'a': tr.get_action(),
                    'r': tr.get_prob() * (tr.get_reward() + self.gamma * tr.get_succ_state().get_value()['r'])
                })

            result.append(
                self.return_max_val(
                    arr))  # nicht den max(array) sondern den für die aktion a. a ist die optimale in Q' ist =>
            # TODO arr-werte mir dazugehörigen aktionen verknüpfuen sodass man nach der optimalen Aktion von Q' filtern kann

        for x in range(len(self.init_set_of_states)):
            self.init_set_of_states[x].set_value(result[x])

    def get_states(self):
        return self.init_set_of_states

    def __init__(self, size, gamma, policy):
        self.size = size
        self.gamma = gamma
        self.policy = policy
        self.init_set_of_states = self.create_set_of_states()
        # initial_distribution_of_states = create_random_initial_state_distribution(init_set_of_states)
        self.initial_state = random.choice(self.init_set_of_states)
        self.initial_state.set_start(True)
        self.end_state = random.choice([state for state in self.init_set_of_states if
                                        (state != self.initial_state)])
        self.end_state.set_end(True)

        # end_state.set_value(1)
        # print(initial_state)
        self.init_set_of_actions = ['up', 'down', 'left', 'right']
        self.init_set_of_transitions = [tr for tr in self.create_set_of_transitions() if
                                        (tr.get_state() != self.end_state)]
        self.init_set_of_transitions.append(Transition(self.end_state, 'exit', State(9, 9)))
        self.init_set_of_transitions_probabilities = self.create_transition_probability()
        self.init_set_of_transitions_probabilities_and_rewards = self.create_transition_reward()
