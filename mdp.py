import numpy as np
from numpy.random import default_rng
import random
from transition import Transition
from state import State
import copy


def delta_x(state1, state2):
    return np.abs(state1.get_x() - state2.get_x())


def delta_y(state1, state2):
    return np.abs(state1.get_y() - state2.get_y())


def return_max_val_abs(arr):
    # #print('max_val arr')
    # for ele in arr:
    #   #print(ele['s'].__dict__, ele['a'], ele['r'])
    if (len(arr) == 0):
        return {'a': 'up', 'r': 0}
    output = arr[0]
    for obj in arr:
        if obj['abs_val'] > output['abs_val']:
            output = obj
    return output


def return_max_val(arr):
    # print('max values: ')
    # for ele in arr:
    #    #print(ele)

    output = arr[0]
    for obj in arr:
        if obj['r'] > output['r']:
            output = obj

    # print('output for state', arr[0]['s'].get_x(), arr[0]['s'].get_y(), output['r'])
    return output


class Mdp:
    def create_set_of_states(self):  # create a quadratic grid world with self.size as its edge length
        result = []
        for i in range(self.size):
            for j in range(self.size):
                result.append(State(i, j))
        return result

    def set_solid_states(self, solid_states_array):
        for state in solid_states_array:
            self.get_state_by_coordinates(state['x'], state['y']).set_solid(True)

    def create_set_of_transitions(self):
        result = []
        for state in self.init_set_of_states:
            for action in [a for a in self.init_set_of_actions if a != 'exit']:
                for succ_state in self.init_set_of_states:
                    t = Transition(state, action, succ_state)
                    if delta_x(t.get_state(), t.get_succ_state()) + delta_y(t.get_state(),
                                                                            t.get_succ_state()) < 2:
                        result.append(Transition(state, action, succ_state))

        return result

    def set_cliff(self, cliff_array):
        self.cliff = cliff_array
        for cliff in cliff_array:
            self.set_cliff_state(cliff)

    def part_of_cliff(self, check_state):
        for state in self.cliff:
            if check_state.get_x() == state['x'] and check_state.get_y() == state['y']:
                return True
        return False

    def get_state_by_coordinates(self, x, y):
        for state in self.init_set_of_states:
            if state.get_x() == x and state.get_y() == y:
                return state
        print("Can't find state with x: " + str(x) + " and y: " + str(y))

    def calculate_prob(self, transition):
        left_border = transition.s.x == 0 or self.get_state_by_coordinates(transition.s.x - 1,
                                                                           transition.s.y).get_solid()
        right_border = transition.s.x == self.size - 1 or self.get_state_by_coordinates(transition.s.x + 1,
                                                                                        transition.s.y).get_solid()
        top_border = transition.s.y == 0 or self.get_state_by_coordinates(transition.s.x,
                                                                          transition.s.y - 1).get_solid()
        bottom_border = transition.s.y == self.size - 1 or self.get_state_by_coordinates(transition.s.x,
                                                                                         transition.s.y + 1).get_solid()

        if transition.a == self.init_set_of_actions[0]:  # action = up
            if transition.s_succ == transition.s:  # stay
                p = self.noise / 4
                if top_border:
                    p += 1 - self.noise
                if bottom_border:
                    p += self.noise / 4
                if left_border or right_border:
                    p += self.noise / 4
                return p
            if transition.s_succ.x == transition.s.x + 1:  # moving right
                return self.noise / 4
            if transition.s_succ.x == transition.s.x - 1:  # moving left
                return self.noise / 4
            if transition.s_succ.y == transition.s.y - 1:  # moving up
                return 1 - self.noise
            if transition.s_succ.y == transition.s.y + 1:  # moving down
                return self.noise / 4

        if transition.a == self.init_set_of_actions[1]:  # down
            if transition.s_succ == transition.s:  # stay
                p = self.noise / 4
                if bottom_border:
                    p += 1 - self.noise
                if top_border:
                    p += self.noise / 4
                if left_border or right_border:
                    p += self.noise / 4
                return p
            if transition.s_succ.x == transition.s.x + 1:  # moving right
                return self.noise / 4
            if transition.s_succ.x == transition.s.x - 1:  # moving left
                return self.noise / 4
            if transition.s_succ.y == transition.s.y - 1:  # moving up
                return self.noise / 4
            if transition.s_succ.y == transition.s.y + 1:  # moving down
                return 1 - self.noise

        if transition.a == self.init_set_of_actions[2]:  # left
            if transition.s_succ == transition.s:  # stay
                p = self.noise / 4
                if left_border:
                    p += 1 - self.noise
                if right_border:
                    p += self.noise / 4
                if top_border or bottom_border:
                    p += self.noise / 4
                return p
            if transition.s_succ.x == transition.s.x + 1:  # moving right
                return self.noise / 4
            if transition.s_succ.x == transition.s.x - 1:  # moving left
                return 1 - self.noise
            if transition.s_succ.y == transition.s.y - 1:  # moving up
                return self.noise / 4
            if transition.s_succ.y == transition.s.y + 1:  # moving down
                return self.noise / 4

        if transition.a == self.init_set_of_actions[3]:  # right
            if transition.s_succ == transition.s:  # stay
                p = self.noise / 4
                if right_border:
                    p += 1 - self.noise
                if left_border:
                    p += self.noise / 4
                if top_border or bottom_border:
                    p += self.noise / 4
                return p
            if transition.s_succ.x == transition.s.x + 1:  # moving right
                return 1 - self.noise
            if transition.s_succ.x == transition.s.x - 1:  # moving left
                return self.noise / 4
            if transition.s_succ.y == transition.s.y - 1:  # moving up
                return self.noise / 4
            if transition.s_succ.y == transition.s.y + 1:  # moving down
                return self.noise / 4

    def create_transition_probability(self):
        result = []
        for t in self.init_set_of_transitions:
            t.set_prob(round(self.calculate_prob(t), 2))
            result.append(t)
        return result

    def transit(self, state):
        state.increase_frequency(1)
        rng = default_rng()
        # print('transit: action result for one state with prob', state.get_x(), state.get_y(), state.get_add_values())
        transitions = []
        prob = []
        # print("last state",state.__dict__)
        if len(state.get_add_values()) == 0:
            random_action = state.get_value()
            random_action['a'] = "exit"
        else:
            random_action = random.choice(state.get_add_values())

        if len(state.get_add_values())>1:
            print('multi add',state.get_add_values())


        for t in [tr for tr in self.init_set_of_transitions_probabilities_and_rewards if
                  tr.get_state() == state and tr.get_action() == random_action['a']]:
            # print(t.__dict__, t.get_succ_state().get_x(), t.get_succ_state().get_y())
            transitions.append(t)
            prob.append(t.get_prob())
        print('probability dist',prob)
        transition = rng.choice(transitions, p=prob, )
        print('transition',transition.get_action())
        # #print('succstate', succ_state.__dict__)
        return {
            'current_state': transition.get_state(),
            'succ_state': transition.get_succ_state(),
            'reward': transition.get_reward()
        }

    def print_prob(self):
        for state in self.init_set_of_states:
            self.transit(state)

    def run_episode_mo(self):
        # print('run episode mooooo')
        max_transition_threshold = 1000
        current_state = self.init_state
        previous_state = {}
        i = 0
        # print('episode is running...')
        discounted_reward = 0
        # states have been set print("episode started")
        print('start', self.return_start().get_x(), self.return_start().get_y())
        while not current_state.is_terminal() and i < max_transition_threshold:
            # new_state = self.transit(current_state)
            # print('new state', new_state['current_state'].__dict__)
            #
            # current_state = new_state['succ_state']
            new_state = self.transit(current_state)
            previous_state = new_state["current_state"]
            current_state = new_state['succ_state']
            discounted_reward += new_state['reward'] * pow(self.gamma, i)
            #discounted_reward *= self.gamma
            i += 1
            print('EPISODE MOMDP: ', i,discounted_reward,current_state.get_x(),current_state.get_y(),current_state.get_value())

        current_state.increase_frequency(1)
        # print('what is the curren state:',current_state.get_value()['r'],pow(self.gamma,i-1),
        # current_state.get_value()['r']*pow(self.gamma,i-1))


        if i == max_transition_threshold:
            print('maximum of transitions')
        if previous_state.get_finish() or i == max_transition_threshold:
            # print('success!', i,discounted_reward )
            return {'iteration': i,
                    'success': True,
                    'discounted_reward': discounted_reward * self.gamma}
        else:
            #  print('failure!', i, discounted_reward)
            return {'iteration': i,
                    'success': False,
                    'discounted_reward': discounted_reward * self.gamma
                    }

    def run_episode_limo(self):
        # print('run episode limooo')
        max_transition_threshold = 1000
        current_state = self.init_state
        i = 0
        # print('episode is running...')
        discounted_reward = 0
        #print('start',self.return_start().get_x(),self.return_start().get_y())
        while not current_state.get_finish() and not self.part_of_cliff(
                current_state) and i < max_transition_threshold:
            new_state = self.transit(current_state)
            # print(current_state.get_add_values())
            current_state = new_state['succ_state']
            discounted_reward += new_state['reward'] * pow(self.gamma, i)
            # discounted_reward *= self.gamma
            i += 1
            print('EPISODE LIMO: ',i,discounted_reward,current_state.get_x(),current_state.get_y(),current_state.get_value())

        discounted_reward += current_state.get_value()['r'] * pow(self.gamma, i)
        #discounted_reward *= self.gamma
        i += 1
        print('episode has ended',discounted_reward)

        current_state.increase_frequency(1)
        # print('what is the curren state:',current_state.get_value()['r'],pow(self.gamma,i-1),
        # current_state.get_value()['r']*pow(self.gamma,i-1))



        if i == max_transition_threshold:
            print('maximum of transitions')
        if current_state.get_finish() or i == max_transition_threshold:
            # print('success!', i, discounted_reward)
            return {'iteration': i,
                    'success': True,
                    'discounted_reward': discounted_reward * self.gamma}
        else:
            # print('failure!', i, discounted_reward)
            return {'iteration': i,
                    'success': False,
                    'discounted_reward': discounted_reward * self.gamma
                    }

    def create_transition_reward(self):
        result = []
        for t in self.init_set_of_transitions_probabilities:
            if self.environment == 'deep-sea-treasure':
                t.set_reward(-1)
            result.append(t)
        # print('function ended')
        return result

    def solve_mdp(self):
        theta = 0.001
        converged = False
        i = 0
        while not converged:
            delta_check = True
            old_states = copy.deepcopy(self.init_set_of_states)
            self.increment_iteration()
            for x in range(len(self.init_set_of_states)):
                delta = np.abs(self.init_set_of_states[x].get_value()['r'] - old_states[x].get_value()['r'])
                delta_check = delta_check and delta < theta
            converged = delta_check and i >= self.size
            i += 1

        return self.init_set_of_states

    def evaluate_policy(self):
        i = 0
        old_states = copy.deepcopy(self.init_set_of_states)
        for state in self.init_set_of_states:
            print(state.get_x(), state.get_y(), state.get_value()['a'])
            state.set_value({'a': state.get_value()['a'], 'r': 0})
        while i < 1:
            for x in range(len(self.init_set_of_states)):
                action = old_states[x].get_value()['a']
                possible_transitions = [tr for tr in self.init_set_of_transitions_probabilities_and_rewards if
                                        tr.get_action() == action and tr.get_state() == (self.init_set_of_states[x])]
                value = 0
                for transition in possible_transitions:
                    print(transition.get_state().get_x(), transition.get_state().get_y(), transition.get_action(),
                          transition.get_reward())
                    value += (transition.get_prob() * (
                            transition.get_reward() + self.gamma * transition.get_succ_state().get_value()['r']))
                print(len(possible_transitions), value)
                self.init_set_of_states[x].set_value({'a': action, 'r': value})
            i += 1

        return self.init_set_of_states

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

    def set_transition_penalty(self, value):  # used to scale transition penalties in deep sea treasure
        for transition in self.init_set_of_transitions_probabilities_and_rewards:
            if not transition.get_state().get_finish():
                transition.set_reward(value)

    def scale_end_states(self, coefficient):  # used to scale the treasures in deep sea treasure
        for state_dict in self.treasures:
            state = self.get_state_by_coordinates(state_dict['x'], state_dict['y'])
            # print("this are the endstates DST ", state_dict,state.__dict__)
            for transition in self.get_transitions_for_state(state):
                if transition.get_state().get_finish():
                    transition.set_reward(state_dict['reward'] * coefficient)

    def clear_states(self):
        for state in self.init_set_of_states:
            state.clear_value()

    def print_treasures(self):
        print('print treasures', self.treasures)

    def set_finish(self, state_dict):
        state = self.get_state_by_coordinates(state_dict['x'], state_dict['y'])
        state.set_finish(True)
        transition = Transition(state, 'exit', State('end', 'end'))
        transition.set_prob(1)
        transition.set_reward(state_dict['reward'])
        self.init_set_of_transitions_probabilities_and_rewards = [tr for tr in
                                                                  self.init_set_of_transitions_probabilities_and_rewards
                                                                  if
                                                                  (tr.get_state() != state)]
        self.init_set_of_transitions_probabilities_and_rewards.append(transition)

    def set_cliff_state(self, cliff_state_dict):
        state = self.get_state_by_coordinates(cliff_state_dict['x'], cliff_state_dict['y'])
        transition = Transition(state, 'exit', State('end', 'end'))
        transition.set_prob(1)
        transition.set_reward(cliff_state_dict['reward'])
        self.init_set_of_transitions_probabilities_and_rewards = [tr for tr in
                                                                  self.init_set_of_transitions_probabilities_and_rewards
                                                                  if
                                                                  (tr.get_state() != state)]
        self.init_set_of_transitions_probabilities_and_rewards.append(transition)

    def set_end(self, state_dict, finish):
        if finish:
            self.set_finish(state_dict)
        else:
            self.set_cliff_state(state_dict)

    def accu_states(self, states, last_state, this_state):
        trans = [tr for tr in self.init_set_of_transitions_probabilities_and_rewards if tr.get_state() == this_state]
        if len(trans) == 0:
            # print('states FAIL BEFORE')
            return []
        trans = [tr for tr in trans if
                 tr.get_prob() >= 0.8 and tr.get_succ_state() != last_state and tr.get_succ_state() != this_state and
                 tr.get_succ_state().get_value()['r'] > this_state.get_value()['r']]
        if this_state.get_finish():
            # print('states SUCCESS', states)
            res = states
            res.append(this_state)
            return res

        if len(trans) == 0:
            # print('states FAIL')
            return []

        max_val = trans[0]
        for tr in trans:
            if tr.get_succ_state().get_value()['r'] > max_val.get_succ_state().get_value()['r']:
                # print('da wurde maximiert')
                max_val = tr
        st = states
        st.append(max_val.get_state())
        return self.accu_states(st, this_state, max_val.get_succ_state())

    def get_transitions_for_state(self, state):
        return [t for t in self.init_set_of_transitions_probabilities_and_rewards if
                (t.get_state() == state)]

    def increment_iteration(self):
        states = [state for state in self.init_set_of_states]
        result = []
        for state in states:
            arr = []
            for action in self.init_set_of_actions:
                possible_transitions = [tr for tr in self.get_transitions_for_state(state) if tr.get_action() == action]
                value = 0
                action_value = {}
                # print('possible trans: ', possible_transitions)
                for transition in possible_transitions:
                    value += (transition.get_prob() * (
                            transition.get_reward() + self.gamma * transition.get_succ_state().get_value()['r']))
                    action_value = {
                        's': transition.get_state(),
                        'a': transition.get_action(),
                        'r': value
                    }
                if len(possible_transitions) > 0:
                    arr.append(action_value)
            # print('this is the max val arr: ', arr)
            max_action_value = return_max_val(arr)
            all_max_val = [v for v in arr if v['r'] == max_action_value['r']]  # nur zwecks visualisierung
            state.set_add_values(all_max_val)  #
            result.append(max_action_value)

        for x in range(len(states)):
            states[x].set_value(result[x])

    def get_selectable_transitions_for_state(self, state):
        arr = []
        for tr in [t for t in self.init_set_of_transitions_probabilities_and_rewards if
                   (t.get_state() == state) and t.get_prob() == 0.8]:
            arr.append({
                's': tr.get_state(),
                'a': tr.get_action(),
                'abs_val': tr.get_succ_state().get_value()['r']
            })
        return arr

    def eval_policy(self):
        result = []
        for state in self.init_set_of_states:
            arr = self.get_selectable_transitions_for_state(state)

            max_val = return_max_val_abs(arr)

            all_max_val = [v for v in arr if (v['abs_val'] == max_val['abs_val'])]  # nur zwecks visualisierung
            # print('this is all max value ', all_max_val)
            state.set_add_values(all_max_val)  #

            result.append(
                max_val)  # nicht den max(array) sondern den für die aktion a. a ist die optimale in Q' ist =>

        for x in range(len(self.init_set_of_states)):
            self.init_set_of_states[x].set_value(
                {

                    'a': result[x]['a'],
                    'r': self.init_set_of_states[x].get_value()['r']
                }
            )

    def set_treasures(self, treasures):

        self.treasures = treasures

    def get_states(self):
        return self.init_set_of_states

    def set_environment(self, environment):
        self.init_set_of_states = self.create_set_of_states()
        self.init_state = {}
        self.cliff = []
        self.environment = environment
        self.init_set_of_actions = ['up', 'down', 'left', 'right', 'exit']
        self.init_set_of_transitions = self.create_set_of_transitions()
        self.init_set_of_transitions_probabilities = self.create_transition_probability()
        self.init_set_of_transitions_probabilities_and_rewards = self.create_transition_reward()

    def set_size(self, size):
        self.size = size

    def __init__(self, size, gamma, policy):
        self.noise = 0.2
        self.size = int(size)
        self.gamma = gamma
        self.policy = policy
        self.init_set_of_states = self.create_set_of_states()
        self.init_state = {}
        self.cliff = []
        self.environment = 'grid-world'
        self.init_set_of_actions = ['up', 'down', 'left', 'right', 'exit']
        self.init_set_of_transitions = self.create_set_of_transitions()
        self.init_set_of_transitions_probabilities = self.create_transition_probability()
        self.init_set_of_transitions_probabilities_and_rewards = self.create_transition_reward()
        self.treasures = []
