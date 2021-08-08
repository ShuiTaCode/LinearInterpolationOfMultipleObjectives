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

        # arr = [tr for tr in arr if tr['prob'] >= 0.8]
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


def run_safe_iteration_main(c):
    mdp2.run_safe_iteration()
    c.delete('all')
    draw_policy(c, xmdp1, 100, scale * 0.7, mdp1.get_states())
    draw_policy(c, canvas_width - size * scale * 0.7, 100, scale * 0.7, mdp2.get_states())
    draw_graph(c, xmdp1, 100, scale * 0.7, mdp1)
    draw_graph(c, canvas_width - size * scale * 0.7, 100, scale * 0.7, mdp2)


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


def compare_states_from_different_mdp(state1, state2):
    # print('compared states', state1, state2)
    return state1.get_x() == state2.get_x() and state1.get_y() == state2.get_y()


def return_min_val(self, arr):
    # print('min_val arr from state',state.__dict__)
    # for ele in arr:
    #     print(ele['s'].__dict__, ele['a'], ele['r'])
    if (len(arr) == 0):
        return {'a': 'up', 'r': 0}
    output = arr[0]
    for obj in arr:
        if obj['r'] < output['r']:
            output = obj
    return output


def run_iteration(self):
    print('iteration wird ausgeführt')
    result = []
    for state in self.init_set_of_states:
        arr = []
        for tr in self.get_transitions_for_state(state):
            arr.append({
                's': tr.get_state(),
                'a': tr.get_action(),
                'r': tr.get_prob() * (tr.get_reward() + self.gamma * tr.get_succ_state().get_value()['r']),
                'succ': tr.get_succ_state(),
                'prob': tr.get_prob(),
                'abs_val': tr.get_succ_state().get_value()['r']
            })

        # arr = [tr for tr in arr if tr['prob'] >= 0.8]
        # if state.get_end() == False and state.get_x() == 0 and state.get_y() == 3:
        #   print('state 0 und 3', state.get_x(), state.get_y())
        #  for value in arr:
        #     print(value['a'], value['r'], value['abs_val'], value['prob'], value['succ'].get_x(),
        #          value['succ'].get_y())
        max_val = {}
        val = self.return_max_val(arr)
        if state.get_end():
            val = self.return_max_val(arr)
        if self.part_of_cliff(state):
            val = self.return_min_val(arr)
        # elif self.positive:
        #    val = self.return_max_val(arr)
        # else:
        #   val = self.return_min_val(arr)

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
