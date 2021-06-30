import numpy as np
from numpy.random import default_rng

from transition import Transition
from state import State


def delta_x(state1, state2):
    return np.abs(state1.get_x() - state2.get_x())


def delta_y(state1, state2):
    return np.abs(state1.get_y() - state2.get_y())


def compare_states_from_different_mdp(state1, state2):
    # print('compared states', state1, state2)
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

    def set_cliff(self, cliff_array):
        self.cliff = cliff_array
        for cliff in cliff_array:
            for state in self.init_set_of_states:
                if state.get_x() == cliff['x'] and state.get_y() == cliff['y']:
                    self.init_set_of_transitions = [tr for tr in self.init_set_of_transitions if
                                                    (tr.get_state() != state)]
                    self.init_set_of_transitions.append(Transition(state, 'exit', State(9, 9)))

        # self.init_set_of_transitions = [tr for tr in self.init_set_of_transitions if
        #                              tr.get_state().get_end() and tr.get_action=='exit'  ]

        # self.init_set_of_transitions = self.create_set_of_transitions()

        self.init_set_of_transitions_probabilities = self.create_transition_probability()
        self.init_set_of_transitions_probabilities_and_rewards = self.create_transition_reward()
        #self.run_iteration()
        # print('reward')
        # for ele in self.init_set_of_transitions_probabilities_and_rewards:
        #    print(ele.get_reward())

    def part_of_cliff(self, check_state):
        for state in self.cliff:
            if check_state.get_x() == state['x'] and check_state.get_y() == state['y']:
                return True
        return False

    def calculate_prob(self, transition):
        left_border = transition.s.x == 0
        right_border = transition.s.x == self.size - 1
        top_border = transition.s.y == 0
        bottom_border = transition.s.y == self.size - 1

        if transition.a == 'exit':
            if transition.s.get_end() or self.part_of_cliff(transition.get_state()):
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

    def transit(self, state):
        rng = default_rng()
        print('transit: action result for one state with prob', state.get_x(), state.get_y())
        succ_states = []
        prob = []
        for t in [tr for tr in self.init_set_of_transitions_probabilities_and_rewards if
                  tr.get_state() == state and tr.get_action() == state.get_value()['a']]:
            print(t.__dict__, t.get_succ_state().get_x(), t.get_succ_state().get_y())
            succ_states.append(t.get_succ_state())
            prob.append(t.get_prob())

        succ_state = rng.choice(succ_states, p=prob, )
        # print('succstate', succ_state.__dict__)
        return succ_state

    def print_prob(self):
        for state in self.init_set_of_states:
            self.transit(state)

    def run_episode(self):
        current_state = self.init_state
        i = 0
        print('episode is running...')
        while not current_state.get_end() and not self.part_of_cliff(current_state):
            new_state = self.transit(current_state)
            print('new state', new_state.__dict__)
            current_state = new_state
            i += 1

        print('number of transitions', i)

        if current_state.get_end():
            return 1
        else:
            return -1

    def create_transition_reward(self):
        result = []
        for t in self.init_set_of_transitions_probabilities:
            if t.get_action() == 'exit':
                if t.get_state().get_end():
                    t.set_reward(1)
                elif self.part_of_cliff(t.get_state()):
                    t.set_reward(-1)
            result.append(t)
        print('function ended')
        return result

    def solve_mdp(self):
        if self.positive:
            while len([s for s in self.init_set_of_states if (s.get_value()['r'] != 0)]) < self.size * self.size - 5:
                self.run_iteration()
        else:
            while len([s for s in self.init_set_of_states if
                       (s.get_value()['r'] != 0)]) < self.size * self.size - 1:
                self.run_iteration()

        return self.init_set_of_states

    def return_max_val(self, arr):
        # print('max_val arr')
        # for ele in arr:
        #   print(ele['s'].__dict__, ele['a'], ele['r'])
        output = arr[0]
        for obj in arr:
            if obj['r'] > output['r']:
                output = obj
        return output

    def return_max_val_abs(self, arr):
        # print('max_val arr')
        # for ele in arr:
        #   print(ele['s'].__dict__, ele['a'], ele['r'])
        output = arr[0]
        for obj in arr:
            if obj['abs_val'] > output['abs_val']:
                output = obj
        return output

    def return_min_val(self, arr):
        #print('min_val arr from state',state.__dict__)
        #for ele in arr:
        #     print(ele['s'].__dict__, ele['a'], ele['r'])
        output = arr[0]
        for obj in arr:
            if obj['r'] < output['r']:
                output = obj
        return output

    def get_value_by_policy(self, arr):
        #  print('das ist das array')
        # for t in arr:
        #    print(t['s'].__dict__, t)
        state_from_arr = arr[0]['s']  # arr hat immer gleichen startpunkt s
        action = ""
        if arr[0]['a'] == 'exit':
            return arr[0]
        for state in self.policy:
            if compare_states_from_different_mdp(state, state_from_arr):
                # print('gefunden', state.get_value()['a'])
                action = state.get_value()['a']

        if action == 'exit':  # wenn policy im entzustand nehme einfach erste lösung
            return arr[0]

        # print('das sind nun all tr einer action', [obj for obj in arr if
        #                                           (obj['a'] == action)])
        return self.return_max_val([obj for obj in arr if
                                    (obj['a'] == action)])

        # print('es ist ein fehler aufgetreten das wurde nicht gefunden: ',action)

    def set_states(self, set_of_states):
        self.init_set_of_states = set_of_states

    def set_start(self, state_dict):
        for state in self.init_set_of_states:
            if state.get_x() == state_dict['x'] and state.get_y() == state_dict['y']:
                state.set_start(True)
                self.init_state = state
            else:
                state.set_start(False)

    def return_start(self):
        return self.init_state

    def set_end(self, state_dict):
        for state in self.init_set_of_states:
            if state.get_x() == state_dict['x'] and state.get_y() == state_dict['y']:
                state.set_end(True)
                self.init_set_of_transitions = [tr for tr in self.init_set_of_transitions if
                                                (tr.get_state() != state)]
                self.init_set_of_transitions.append(Transition(state, 'exit', State(9, 9)))
            else:
                state.set_end(False)

        self.init_set_of_transitions_probabilities = self.create_transition_probability()
        self.init_set_of_transitions_probabilities_and_rewards = self.create_transition_reward()
        #self.run_iteration()

    def run_safe_iteration(self):
        # print('transformation wird ausgeführt')
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
        # print('result length', len(result))
        for x in range(len(self.init_set_of_states)):
            # print('jojojo',result[x]['s'].__dict__,result[x])
            self.init_set_of_states[x].set_value(result[x])

    def accu_reward(self, lastState, thisState):
        trans = [tr for tr in self.init_set_of_transitions_probabilities_and_rewards if tr.get_state() == thisState]
        if len(trans) == 0:
            print('FAIL BEFORE')
            return thisState.get_value()
        trans = [tr for tr in trans if
                 tr.get_prob() >= 0.8 and tr.get_succ_state() != lastState and tr.get_succ_state() != thisState and
                 tr.get_succ_state().get_value()['r'] > thisState.get_value()['r']]
        if thisState.get_end():
            print('SUCCESS')
            return thisState.get_value()['r']

        if len(trans) == 0:
            print('FAIL')
            return thisState.get_value()['r']

        max_val = trans[0]
        for tr in trans:
            if tr.get_succ_state().get_value()['r'] > max_val.get_succ_state().get_value()['r']:
                print('da wurde maximiert')
                max_val = tr

        return max_val.get_state().get_value()['r'] + self.accu_reward(thisState, max_val.get_succ_state())

    def accu_states(self, states, lastState, thisState):
        trans = [tr for tr in self.init_set_of_transitions_probabilities_and_rewards if tr.get_state() == thisState]
        if len(trans) == 0:
            print('states FAIL BEFORE')
            return []
        trans = [tr for tr in trans if
                 tr.get_prob() >= 0.8 and tr.get_succ_state() != lastState and tr.get_succ_state() != thisState and
                 tr.get_succ_state().get_value()['r'] > thisState.get_value()['r']]
        if thisState.get_end():
            print('states SUCCESS', states)
            res = states
            res.append(thisState)
            return res

        if len(trans) == 0:
            print('states FAIL')
            return []

        max_val = trans[0]
        for tr in trans:
            if tr.get_succ_state().get_value()['r'] > max_val.get_succ_state().get_value()['r']:
                print('da wurde maximiert')
                max_val = tr
        st = states
        st.append(max_val.get_state())
        return self.accu_states(st, thisState, max_val.get_succ_state())

    def eval_policy(self):
        result = []
        for state in self.init_set_of_states:
            arr = []
            for tr in [t for t in self.init_set_of_transitions_probabilities_and_rewards if
                       (t.get_state() == state)]:
                arr.append({
                    's': tr.get_state(),
                    'a': tr.get_action(),
                    'r': tr.get_prob() * (tr.get_reward() + self.gamma * tr.get_succ_state().get_value()['r']),
                    'succ': tr.get_succ_state(),
                    'prob': tr.get_prob(),
                    'abs_val': tr.get_succ_state().get_value()['r']
                })

            arr = [tr for tr in arr if tr['prob'] >= 0.8]
            # if state.get_end() == False:
            # print('array der nicht end states eval', state.get_x(), state.get_y())
            # for value in arr:
            #   print(value['a'], value['r'], value['abs_val'], value['prob'], value['succ'].get_x(),
            #        value['succ'].get_y())

            max_val = self.return_max_val_abs(arr)

            all_max_val = [v for v in arr if (v['abs_val'] == max_val['abs_val'])]  # nur zwecks visualisierung
            state.set_add_values(all_max_val)  #

            result.append(
                max_val)  # nicht den max(array) sondern den für die aktion a. a ist die optimale in Q' ist =>
            # TODO arr-werte mit dazugehörigen aktionen verknüpfen sodass man nach der optimalen Aktion von Q' filtern kann

        for x in range(len(self.init_set_of_states)):
            self.init_set_of_states[x].set_value(
                {

                    'a': result[x]['a'],
                    'r': self.init_set_of_states[x].get_value()['r']
                }
            )
        # print('neue policy', {

    #
    #               'a': result[x]['a'],
    #              'r': self.init_set_of_states[x].get_value()['r']
    #         })

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
                    'r': tr.get_prob() * (tr.get_reward() + self.gamma * tr.get_succ_state().get_value()['r']),
                    'succ': tr.get_succ_state(),
                    'prob': tr.get_prob(),
                    'abs_val': tr.get_succ_state().get_value()['r']
                })

            arr = [tr for tr in arr if tr['prob'] >= 0.8]
            # if state.get_end() == False and state.get_x() == 0 and state.get_y() == 3:
            #   print('state 0 und 3', state.get_x(), state.get_y())
            #  for value in arr:
            #     print(value['a'], value['r'], value['abs_val'], value['prob'], value['succ'].get_x(),
            #          value['succ'].get_y())
            max_val = {}
            if state.get_end():
                val = self.return_max_val(arr)
            if self.part_of_cliff(state) :
                val = self.return_min_val(arr)
            elif self.positive:
                val = self.return_max_val(arr)
            else:
                val = self.return_min_val(arr)

            max_val = self.return_max_val_abs(arr)

            all_max_val = [v for v in arr if (v['abs_val'] == max_val['abs_val'])]  # nur zwecks visualisierung

            #   if state.get_end() == False and state.get_x() == 0 and state.get_y() == 3:
            # print('state03 all max')
            # for value in all_max_val:
            #   print(value['a'], value['r'], value['abs_val'], value['prob'], value['succ'].get_x(),
            #        value['succ'].get_y())

            # all_val = [v for v in arr if (v['r'] == val['r'])]  # nur zwecks visualisierung
            # state.set_add_values(all_max_val)  #
            state.set_add_values(all_max_val)  #

            # state.set_add_values(all_val)  #

            # if max_val != {}:
            #   all_val = [v for v in arr if (v['r'] == max_val['r'])]  # nur zwecks visualisierung
            #  # state.set_add_values(all_max_val)  #

            # state.set_add_values([max_val])  #

            result.append(
                val)  # nicht den max(array) sondern den für die aktion a. a ist die optimale in Q' ist =>
            # TODO arr-werte mit den dazugehörigen aktionen verknüpfuen sodass man nach der optimalen Aktion von Q' filtern kann

        for x in range(len(self.init_set_of_states)):
            self.init_set_of_states[x].set_value(result[x])

    def get_states(self):
        return self.init_set_of_states

    def __init__(self, size, gamma, policy, positive):
        self.positive = positive
        self.size = size
        self.gamma = gamma
        self.policy = policy
        self.init_set_of_states = self.create_set_of_states()
        self.init_state = {}
        self.cliff = []
        # initial_distribution_of_states = create_random_initial_state_distribution(init_set_of_states)
        # self.initial_state = random.choice(self.init_set_of_states)
        # self.initial_state.set_start(True)
        # self.end_state = random.choice([state for state in self.init_set_of_states if state != self.initial_state])
        # self.end_state.set_end(True)

        # end_state.set_value(1)
        # print(initial_state)
        self.init_set_of_actions = ['up', 'down', 'left', 'right']
        # self.init_set_of_transitions = [tr for tr in self.create_set_of_transitions() if
        #                                 (tr.get_state() != self.end_state)]
        # self.init_set_of_transitions.append(Transition(self.end_state, 'exit', State(9, 9)))
        self.init_set_of_transitions = self.create_set_of_transitions()
        # self.init_set_of_transitions_probabilities = self.create_transition_probability()
        # self.init_set_of_transitions_probabilities_and_rewards = self.create_transition_reward()
