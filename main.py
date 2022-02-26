import copy
import math
import random
from tkinter import *

# import matplotlib
import numpy
import numpy as np
from matplotlib import pyplot as plt

from mdp import Mdp

from LIMOconfig import LIMOconfiguration


def start_cliff_world():
    var_gamma_calc = int(var_gamma.get()) / 100.0
    alpha = int(var_alpha.get()) / 100.0
    mdp1.__init__(int(var_size.get()), var_gamma_calc, [],config.NOISE_FACTOR)
    mdp2.__init__(int(var_size.get()), var_gamma_calc, [],config.NOISE_FACTOR)
    mdp3.__init__(int(var_size.get()), var_gamma_calc, [],config.NOISE_FACTOR)
    mdp4.__init__(int(var_size.get()), var_gamma_calc, [],config.NOISE_FACTOR)
    mdp1.set_environment('cliff-world')
    mdp2.set_environment('cliff-world')
    mdp3.set_environment('cliff-world')
    mdp4.set_environment('cliff-world')

    start = {'x': int(var_size.get()) - 2, 'y': int(var_size.get()) - 1}
    end = {'x': int(var_size.get()) - 2, 'y': 0}

    mdp1.set_start(start)
    cliff = create_cliff(-1)
    cliff_pos_mdp = create_cliff(0)
    mdp1.set_cliff(cliff)

    mdp2.set_start(start)
    mdp2.set_finish({'x': end['x'], 'y': end['y'], 'reward': 1})
    mdp2.set_cliff(cliff_pos_mdp)

    mdp3.set_start(start)
    mdp3.set_finish({'x': end['x'], 'y': end['y'], 'reward': 1})
    mdp3.set_cliff(cliff)
    mdp4.set_start(start)
    set_rewards_for_mo_mdp(alpha)

    draw_square(c3, 2000, 2000, 5, 5, 'white')
    draw_graphs_and_policies()
    draw_heatmaps()


def get_solid_states_for_deep_sea(end_state_arr):
    res = []
    for state in end_state_arr:
        i = state.get_y() + 1  # position beneath the treasure state
        while i < int(var_size.get()):  # until the bottom of the gridworld
            res.append({'x': state.get_x(), 'y': i})
            i += 1
    return res


def create_cliff(reward):
    cliff = []
    for i in range(int(var_size.get())):
        cliff.append({
            'x': int(var_size.get()) - 1,
            'y': i,
            'reward': reward
        })
    return cliff


def get_treasures_for_deep_sea():
    var_size_calc = int(var_size.get())
    result = [{
        'x': 0,
        'y': 1,
        'reward': 1
    }]
    start_depth = 1
    for i in range(var_size_calc - 1):
        x = i + 1
        random_step = random.choice(range(var_size_calc - start_depth))
        y = start_depth + random_step
        start_depth += random_step
        result.append({
            'x': x,
            'y': y,
            'reward': x * y * 4,
            'magic': x * y * 4
        })
    return result


def set_rewards_for_limo(alpha):  # only call after
    environment = env_option.get()
    if environment == 'deep-sea-treasure':
        # mdp3.scale_end_states(1 - alpha)
        mdp3.set_transition_penalty(-1 * alpha)


def set_rewards_for_mo_mdp(alpha):
    environment = env_option.get()
    mdp4.clear_states()

    if environment == 'cliff-world':
        cliff = create_cliff(-1 * alpha)
        end = {'x': int(var_size.get()) - 2, 'y': 0}
        mdp4.set_finish({'x': end['x'], 'y': end['y'], 'reward': 1 - alpha})
        mdp4.set_cliff(cliff)
        mdp4.solve_mdp()

    if environment == 'deep-sea-treasure':
        mdp4.scale_end_states(1 - alpha)
        mdp4.set_transition_penalty(-1 * alpha)
        mdp4.solve_mdp()


def get_initial_treasures_for_DST():
    treasure_scale = 4
    treasure_y_coordinates = [1, 2, 2, 3, 3, 5, 5, 6, 7, 8]
    x = 0
    res = []
    for index in range(int(var_size.get())):
        res.append({
            'x': x,
            'y': treasure_y_coordinates[index],
            'reward': (x + treasure_y_coordinates[index]) * treasure_scale

        })
        x += 1
    return res


def start_deep_sea_treasure():
    var_gamma_calc = int(var_gamma.get()) / 100
    alpha = int(var_alpha.get()) / 100.0
    mdp1.__init__(int(var_size.get()), var_gamma_calc, [],config.NOISE_FACTOR)
    mdp2.__init__(int(var_size.get()), var_gamma_calc, [],config.NOISE_FACTOR)
    mdp3.__init__(int(var_size.get()), var_gamma_calc, [],config.NOISE_FACTOR)
    mdp4.__init__(int(var_size.get()), var_gamma_calc, [],config.NOISE_FACTOR)
    mdp1.set_environment('deep-sea-treasure')
    mdp2.set_environment('cliff-world')  # cliffworld environment to reduce boilerplate for simple goal states MDP
    mdp3.set_environment('deep-sea-treasure')
    mdp4.set_environment('deep-sea-treasure')
    start = {'x': 0, 'y': 0}
    mdp1.set_start(start)
    mdp2.set_start(start)
    mdp3.set_start(start)
    mdp4.set_start(start)
    # treasures = get_treasures_for_deep_sea()
    treasures = get_initial_treasures_for_DST()
    for treasure in treasures:
        mdp2.set_finish(treasure)
        mdp3.set_finish(treasure)
        mdp4.set_finish(treasure)

    mdp4.set_treasures(treasures)

    for treasure in copy.deepcopy(treasures):
        treasure[
            'reward'] = 0  # no reward/punishment when the episode ends => urges the agent to take the nearest treasure
        mdp1.set_finish(treasure)

    solid_states = get_solid_states_for_deep_sea([state for state in mdp2.get_states() if state.get_finish()])
    mdp1.set_solid_states(solid_states)
    mdp2.set_solid_states(solid_states)
    mdp3.set_solid_states(solid_states)
    mdp4.set_solid_states(solid_states)

    set_rewards_for_mo_mdp(alpha)
    mdp4.increment_iteration()

    draw_square(c3, 2000, 2000, 5, 5, 'white')
    draw_graphs_and_policies()
    draw_heatmaps()


def draw_square(c, x, y, w, h, color):
    c.create_line(x, y, x + w, y, fill=color, width=3)
    c.create_line(x + w, y, x + w, y + h, fill=color, width=3)
    c.create_line(x + w, y + h, x, y + h, fill=color, width=3)
    c.create_line(x, y + h, x, y, fill=color, width=3)


def draw_triangle(c, x, y, w, h, action, reward, color):
    stretch_width = 0.5  # stretch and compress in width of arrow
    delta = 30  # distance from the center of the square
    stretch_height = 0.5  # stretch and compress in height of arrow
    if reward < 999:
        if action == 'up':
            c.create_line(x, y - delta, x + stretch_width * w, y - delta, fill=color, width=3)
            c.create_line(x + stretch_width * w, y - delta, x, y - stretch_width * h * stretch_height - delta,
                          fill=color,
                          width=3)
            c.create_line(x, y - stretch_width * h * stretch_height - delta, x - stretch_width * w, y - delta,
                          fill=color,
                          width=3)
            c.create_line(x - stretch_width * w, y - delta, x, y - delta, fill=color, width=3)
            return
        if action == 'down':
            c.create_line(x, y + delta, x + stretch_width * w, y + delta, fill=color, width=3)
            c.create_line(x + stretch_width * w, y + delta, x, y + stretch_width * h * stretch_height + delta,
                          fill=color,
                          width=3)
            c.create_line(x, y + stretch_width * h * stretch_height + delta, x - stretch_width * w, y + delta,
                          fill=color,
                          width=3)
            c.create_line(x - stretch_width * w, y + delta, x, y + delta, fill=color, width=3)
            return
        if action == 'left':
            c.create_line(x - delta, y, x - delta, y - stretch_width * h, fill=color, width=3)
            c.create_line(x - delta, y - stretch_width * h, x - delta - stretch_width * w * stretch_height, y,
                          fill=color,
                          width=3)
            c.create_line(x - stretch_width * w * stretch_height - delta, y, x - delta, y + stretch_width * h,
                          fill=color,
                          width=3)
            c.create_line(x - delta, y + stretch_width * h, x - delta, y, fill=color, width=3)
            return
        if action == 'right':
            c.create_line(x + delta, y, x + delta, y - stretch_width * h, fill=color, width=3)
            c.create_line(x + delta, y - stretch_width * h, x + stretch_width * w * stretch_height + delta, y,
                          fill=color,
                          width=3)
            c.create_line(x + stretch_width * w * stretch_height + delta, y, x + delta, y + stretch_width * h,
                          fill=color,
                          width=3)
            c.create_line(x + delta, y + stretch_width * h, x + delta, y, fill=color, width=3)
            return


def validate_input(P):
    if (P.isdigit() and int(P) <= 100) or P == "":
        return True
    else:
        return False


def validate_input_size(P):
    if (P.isdigit() and int(P) <= 10) or P == "":
        return True
    else:
        return False


def draw_graph(c, x, y, mdp):
    draw_square(c, 2000, 2000, 5, 5, 'white')
    temp_init_state = {}
    end_states = []
    solid_states = []
    for s in mdp.get_states():
        if s.get_solid():
            solid_states.append(s)
        elif s.get_start():
            temp_init_state = s
        elif s.get_finish() or mdp.part_of_cliff(s):
            end_states.append(s)

        draw_square(c, x + s.x * scale, y + s.y * scale, scale, scale, 'black')
        c.create_text(x + s.x * scale + scale / 3, y + s.y * scale + scale / 2, text=round(s.get_value()['r'], 3),
                      anchor='nw',
                      font='TkMenuFont', fill='black')
    draw_square(c, x + temp_init_state.get_x() * scale, y + temp_init_state.get_y() * scale, scale, scale, 'blue')
    c.create_text(x + temp_init_state.x * scale, y + temp_init_state.y * scale, text='Start', anchor='nw',
                  font='TkMenuFont', fill='blue')

    for state in solid_states:
        c.create_rectangle(x + state.x * scale, y + state.y * scale, x + state.x * scale + scale,
                           y + state.y * scale + scale, fill="grey", outline='black')
        # draw_square(c, x + state.x * scale, y + state.y * scale, scale, scale, 'grey')

    for temp_end_state in end_states:
        if temp_end_state.get_value()['r'] >= 0:
            draw_square(c, x + temp_end_state.x * scale, y + temp_end_state.y * scale, scale, scale, 'green')
            c.create_text(x + temp_end_state.x * scale, y + temp_end_state.y * scale, text='End ', anchor='nw',
                          font='TkMenuFont', fill='green')
        else:
            draw_square(c, x + temp_end_state.x * scale, y + temp_end_state.y * scale, scale, scale, 'red')
            c.create_text(x + temp_end_state.x * scale, y + temp_end_state.y * scale, text='End ', anchor='nw',
                          font='TkMenuFont', fill='red')
        # draw_square(c, x + temp_end_state.x * scale, y + temp_end_state.y * scale, scale, scale, 'green')

    # c.create_text(x + temp_end_state.x * scale + scale / 2, y + temp_end_state.y * scale + scale / 2,
    #              text=temp_end_state.get_value()['r'], anchor='nw',
    #             font='TkMenuFont', fill='black')


def draw_policy(c, x, y, set_of_states):
    for s in set_of_states:
        if len(s.get_add_values()) < 4:
            for max_value in s.get_add_values():
                draw_triangle(c, x + s.x * scale + 0.5 * scale, y + s.y * scale + 0.5 * scale, 0.5 * scale, 0.5 * scale,
                              max_value['a'], s.value['r'], 'grey')


def draw_heatmap(x, y, set_of_states):
    for s in set_of_states:
        c3.create_rectangle(x + s.x * scale, y + s.y * scale, x + s.x * scale + scale,
                            y + s.y * scale + scale, fill=freq_to_color(s.get_frequency()), )

        c3.create_text(x + s.x * scale + scale / 3, y + 20 + s.y * scale + scale / 2,
                       text=s.get_frequency(),
                       anchor='nw',
                       font='TkMenuFont', fill='yellow')
        # print(
        #     str(s.get_x()) + "/" + str((int(var_size.get()) - 1) - s.get_y()) + "/" + str(s.get_x() + 0.5) + "/" + str(
        #         (int(var_size.get()) - 1) - s.get_y() + 0.5) + "/" + str(
        #         s.get_frequency()) + "/" + str(
        #         255 - math.floor(255 * (s.get_frequency() / (s.get_frequency() + 25)))) + ",")


def print_heatmap(mdp):
    for s in mdp.get_states():
        print(
            str(s.get_x()) + "/" + str((int(var_size.get()) - 1) - s.get_y()) + "/" + str(s.get_x() + 0.5) + "/" + str(
                (int(var_size.get()) - 1) - s.get_y() + 0.5) + "/" + str(
                s.get_frequency()) + "/" + str(
                255 - math.floor(255 * (s.get_frequency() / (s.get_frequency() + 25)))) + ",")


def freq_to_color(freq):  # calculates the color of a state in the heatmap
    c = 25.0  # prevents division by zero if freq == 0
    rgba_hex = '#{:02x}{:02x}{:02x}'.format(120, 0, 255 - math.floor(255 * (freq / (freq + c))))
    return rgba_hex


def run_iteration():  # runs the Value Iteration algorithm for all relevant MDPs
    mdp1.increment_iteration()
    mdp2.increment_iteration()
    mdp4.increment_iteration()
    draw_graphs_and_policies()


def calc_lin_combination(alpha):  # calculates the linear combination based on alpha
    mdp1_states = mdp1.get_states()
    mdp2_states = mdp2.get_states()
    mdp3_states = mdp3.get_states()
    end_states = []

    for i in range(len(mdp3_states)):  # changes only value of the state not it's role (e.g is_end() oder part_of_cliff)
        new_state = mdp3_states[i]
        new_value = {'a': new_state.get_value()['a'],
                     'r': mdp1_states[i].get_value()['r'] * alpha + mdp2_states[i].get_value()['r'] * (1 - alpha)}
        if mdp3_states[i].get_finish():
            end_states.append(mdp3_states[i])
        new_state.set_value(new_value)

    mdp3.eval_policy()


def solve_mdps():  # solves all MDPs including the MOMDP scalarized with the preference defined through alpha
    alpha = int(var_alpha.get()) / 100.0
    mdp1.solve_mdp()
    mdp2.solve_mdp()
    set_rewards_for_mo_mdp(alpha)
    mdp4.solve_mdp()
    draw_graphs_and_policies()


def draw_graphs_and_policies(): # draws the Gridworld-visualisation of the MDPs, value-functions as well as their
    # greedy policies
    y_mdp2 = int(var_size.get()) * scale + padding # the start y coordinate of the lower MDPs
    c1.delete('all')
    c2.delete('all')
    draw_policy(c1, xmdp1, ymdp1, mdp1.get_states())
    draw_policy(c1, xmdp2, y_mdp2, mdp2.get_states())
    draw_policy(c2, xmdp3, ymdp3, mdp3.get_states())
    draw_policy(c2, xmdp3, y_mdp2, mdp4.get_states())
    draw_graph(c1, xmdp1, ymdp1, mdp1)
    draw_graph(c1, xmdp2, y_mdp2, mdp2)
    draw_graph(c2, xmdp3, ymdp3, mdp3)
    draw_graph(c2, xmdp3, y_mdp2, mdp4)
    render_scrollbars()


def set_lin_combination():
    alpha = int(var_alpha.get()) / 100.0
    mdp1.solve_mdp()
    mdp2.solve_mdp()
    calc_lin_combination(alpha)
    set_rewards_for_mo_mdp(alpha)
    mdp4.solve_mdp()
    draw_graphs_and_policies()


def clear_canvas():
    c1.delete('all')
    c2.delete('all')


def graph(mdp, is_limo, label):
    exp_ret_array = []
    act_ret_array = []
    exp = ""
    act = ""
    for x in alpha_definition_set:
        res = run_episode(mdp, is_limo, x)
        exp_ret_array.append(res['exp_ret'])
        act_ret_array.append(res['act_ret'])
        exp = exp + "(" + str(x) + "," + str(res['exp_ret']) + ") "
        act = act + "(" + str(x) + "," + str(res['act_ret']) + ") "

    # print('expected discounted return: ', exp)
    # print('actual discounted return: ', act)
    plot_data(alpha_definition_set, exp_ret_array, act_ret_array, label)


def plot_data(definition_set, plot1, plot2, headline):
    fig = plt.figure()
    fig.suptitle(headline, fontsize=16)
    ax1 = fig.add_subplot(221)
    ax1.scatter(definition_set, plot1, s=10, c='b', marker="o", label='expected return')
    ax1.scatter(definition_set, plot2, s=10, c='#add8e6', marker="s", label='actual return')
    plt.legend(loc='upper right')
    plt.show()


def clear_heat_map(mdp):
    for state in mdp.get_states():
        state.set_frequency(0)


def clear_heat_maps():
    clear_heat_map(mdp3)
    clear_heat_map(mdp4)
    draw_heatmaps()


def run_single_preference(alpha, multiple):
    set_lin_combination()
    y_heat_map_2 = int(var_size.get()) * scale + padding
    if alg_option.get() == 'LIMO':
        clear_heat_map(mdp3)
        res = run_episode(mdp3, 'LIMO', alpha)
        if not multiple:
            draw_heatmap(0, 0, mdp3.get_states())
    else:
        clear_heat_map(mdp4)
        res = run_episode(mdp4, 'MO', alpha)
        if not multiple:
            draw_heatmap(0, y_heat_map_2, mdp4.get_states())

    return res


def draw_heatmaps():
    y_heat_map_2 = int(var_size.get()) * scale + padding
    draw_heatmap(0, 0, mdp3.get_states())
    draw_heatmap(0, y_heat_map_2, mdp4.get_states())


def run_multiple_preferences():
    label = alg_option.get() + '-plot for ' + var_number_of_episodes.get() + '  episodes in ' + env_option.get() + ' Environment'
    exp_ret_array = []
    act_ret_array = []
    exp = ""
    act = ""
    iteration = ""
    for x in alpha_definition_set:
        res = run_single_preference(x, True)
        exp_ret_array.append(res['exp_ret'])
        act_ret_array.append(res['act_ret'])
        exp = exp + "(" + str(x) + "," + str(res['exp_ret']) + ") "
        act = act + "(" + str(x) + "," + str(res['act_ret']) + ") "
        iteration = iteration + "(" + str(x) + "," + str(res['count']) + ") "

    print('expected discounted return: ', exp)
    print('actual discounted return: ', act)
    print('iteration: ', iteration)
    plot_data(alpha_definition_set, exp_ret_array, act_ret_array, label)


def print_policy(mdp):
    for state in mdp.get_states():
        print(state.__dict__['x'], state.__dict__['y'], state.__dict__['add_values'])


def print_evaluation(alpha, pos_reward, neg_reward, count, pos_count):
    print('evaluation of episodes: ')
    print('success-rate pos and iterations', '(' + str(alpha) + ',' + str(pos_count) + ')',
          '(' + str(alpha) + ',' + str(np.median(count)) + ')')
    print('complete reward measured',
          '(' + str(alpha) + ',' + str(round(float(np.mean(pos_reward + neg_reward)), 3)) + ')')
    print('expected reward MOMDP',
          '(' + str(alpha) + ',' + str(round(float(mdp4.return_start().get_value()['r']), 3)) + ')')
    print('expected reward LIMO',
          '(' + str(alpha) + ',' + str(round(float(mdp3.return_start().get_value()['r']), 3)) + ')')
    print('median of transitions: ', np.median(count))


def run_episode(mdp, mode, alpha):
    # print('run episodes for alpha=', alpha)
    pos_count = 0
    neg_count = 0
    pos_reward = []
    neg_reward = []
    count = []
    is_limo = mode == 'LIMO'
    if not is_limo:
        set_rewards_for_mo_mdp(alpha)

    clear_heat_map(mdp)
    calc_lin_combination(alpha)
    # print('exp start: ',mdp.return_start().get_value()['r'])
    # print('policy LIMO')
    # print_policy(mdp3)
    set_rewards_for_limo(alpha)

    for i in range(int(var_number_of_episodes.get())):
        # print('episode ', i, ' of ', var_number_of_episodes.get())
        if is_limo:
            res = mdp3.run_episode_limo()

        else:
            # set_rewards_for_mo_mdp(mdp,alpha)
            # print('policy MOMDP')
            # print_policy(mdp4)
            res = mdp4.run_episode_mo()

        if res['success']:
            pos_count += 1
            count.append(res['iteration'])
            pos_reward.append(res['discounted_reward'])
        else:
            neg_count += 1
            count.append(res['iteration'])
            # print('negative measured discounted reward: ' + str(res['discounted_reward']))
            neg_reward.append(res['discounted_reward'])

    # print_evaluation(alpha, pos_reward, neg_reward, count, pos_count)

    return {
        'exp_ret': round(float(mdp.return_start().get_value()['r']), 3),
        'act_ret': round(float(np.mean(pos_reward + neg_reward)), 3),
        'pos': pos_count,
        'neg': neg_count,
        'count': np.median(count),
        'pos_reward': pos_reward,
        'neg_reward': neg_reward,
    }


def plot_episode_count(episode_data, fig, alpha):
    # x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00]
    count = []
    for episode in episode_data:
        count.append(numpy.median(episode['count']))

    ax1 = fig.add_subplot(223)

    ax1.scatter(alpha, count, s=10, c='b', marker="s", label='number of transitions')
    plt.legend(loc='upper left')


def plot_episode_graph(episode_data, fig, alpha):
    # x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00]
    pos = []
    neg = []
    for episode in episode_data:
        pos.append(episode['pos'])
        neg.append(episode['neg'])

    ax1 = fig.add_subplot(222)

    ax1.scatter(alpha, pos, s=10, c='b', marker="s", label='pos')
    ax1.scatter(alpha, neg, s=10, c='r', marker="o", label='neg')
    plt.legend(loc='upper left')


def plot_graph(mdp1_data, mdp2_data, episode_data, fig, alpha):
    # print('what is episode_data', episode_data)

    pos_data = []
    neg_data = []
    for episode in episode_data:
        pos_data.append(numpy.mean(episode['pos_reward']))
        neg_data.append(numpy.mean(episode['neg_reward']))

    # x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00]
    # print('size',len(mdp1_data),len(mdp2_data),len(x))
    # fig = plt.figure()
    ax1 = fig.add_subplot(221)

    ax1.scatter(alpha, mdp2_data, s=10, c='b', marker="o", label='right Mdp')
    ax1.scatter(alpha, pos_data, s=10, c='#add8e6', marker="s", label='pos_data Mdp')
    ax1.scatter(alpha, mdp1_data, s=10, c='r', marker="o", label='left Mdp')
    ax1.scatter(alpha, neg_data, s=10, c='#f7a6a8', marker="s", label='neg_data Mdp')
    # ax1.scatter(x, sum, s=10, c='g', marker="o", label='both')
    plt.legend(loc='upper right')


def plot_graph_from_data(expected_return_array, measured_return_array, fig, alpha, title):
    # print('what is episode_data', episode_data)

    all_data = []
    for episode in measured_return_array:
        all_data.append(numpy.mean(episode['pos_reward'] + episode['neg_reward']))  # pos und neg data werden
        # concatiniert vorm dem durchschnitt
    fig.suptitle(title, fontsize=16)
    ax1 = fig.add_subplot(221)
    ax1.scatter(alpha, expected_return_array, s=10, c='b', marker="o", label='expected return')
    ax1.scatter(alpha, all_data, s=10, c='#add8e6', marker="s", label='actual return')
    plt.legend(loc='upper right')


def plot_limo_graph(expected_limo, episode_data, fig, alpha):
    # print('what is episode_data', episode_data)

    all_data = []
    for episode in episode_data:
        all_data.append(numpy.mean(episode['pos_reward'] + episode['neg_reward']))

    # x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00]
    # print('size',len(mdp1_data),len(mdp2_data),len(x))
    # fig = plt.figure()
    ax1 = fig.add_subplot(221)

    ax1.scatter(alpha, expected_limo, s=10, c='b', marker="o", label='expected return')
    ax1.scatter(alpha, all_data, s=10, c='#add8e6', marker="s", label='actual return')
    plt.legend(loc='upper right')


def create_definition_set(size):  # creates a discrete interval between 0 and 1 with a 1/size step
    step = 1.0 / size
    counter = 0
    result_set = [0]
    while len(result_set) <= size:
        counter += step
        counter = round(counter, 4)
        result_set.append(counter)
    return result_set


# def resize_mpds(can1, can2):
#     length = int(var_size.get()) * 120
#     can1.__init__(canvas_frame, width=length, height=length, background='white')
#     can1.grid(row=2, column=0, sticky="w")
#
#     can2.__init__(canvas_frame_2, width=length, height=length, background='white')
#     can2.grid(row=2, column=1, sticky="w")


def environment_change_event():
    var_alpha.set('0')
    print('environment changed: ', env_option.get())
    c3.delete('all')
    option = env_option.get()
    if option == 'cliff-world':
        start_cliff_world()
    if option == 'deep-sea-treasure':
        start_deep_sea_treasure()


def print_latex_for_state(state, mdp):
    action = str(state.get_value()['a'])
    if state.get_finish() or mdp.part_of_cliff(state):
        action = 'exit'
    print(str(state.get_x() + 0.5) + "/" + str((int(var_size.get()) - 1) - state.get_y() + 0.5) + "/" + str(
        round(state.get_value()['r'], 3)) + "/" + action + ",")


def print_heatmap_latex_tikzpicture_CW(mdp):
    boiler_start = '\\begin{tikzpicture} \n' \
                   '\\foreach \\xsquare/\\ysquare/\\x/\\y/\\freq/\\col ' \
                   'in { '

    boiler_end = '} \n { \n' \
                 '\\fill[fill={rgb:red,50;green,50;blue,\\col}](\\xsquare,\\ysquare) rectangle (\\xsquare +1,\\ysquare+1);' \
                 '\\draw[white] (\\x,\\y) node {\\scriptsize\\freq};' \
                 '} \n' \
                 '\draw[step=1cm,black,very thin] (0,0) grid (5,5); \n' \
                 '\\draw[step=1cm,red,thick] (4,4) grid (5,5); \n' \
                 '\\draw[step=1cm,red,thick] (4,3) grid (5,4); \n' \
                 '\\draw[step=1cm,red,thick] (4,2) grid (5,3); \n' \
                 '\\draw[step=1cm,red,thick] (4,1) grid (5,2); \n' \
                 '\\draw[step=1cm,red,thick] (4,0) grid (5,1); \n' \
                 '\\draw[step=1cm,green,thick] (3,4) grid (4,5); \n' \
                 '\\end{tikzpicture}'

    print(boiler_start)
    print_heatmap(mdp)
    print(boiler_end)


def print_heatmap_latex_tikzpicture_DST(mdp):
    boiler_start = '\\begin{tikzpicture} \n' \
                   '\\foreach \\xsquare/\\ysquare/\\x/\\y/\\freq/\\col ' \
                   'in { '

    boiler_end = '} \n { \n' \
                 '\\fill[fill={rgb:red,50;green,50;blue,\\col}](\\xsquare,\\ysquare) rectangle (\\xsquare +1,\\ysquare+1);' \
                 '\\draw[white] (\\x,\\y) node {\\scriptsize\\freq};' \
                 '} \n' \
                 '\\fill[gray](0,0) rectangle (1,1); \n' \
                 '\\fill[gray](1,0) rectangle (2,1); \n' \
                 '\\fill[gray](2,0) rectangle (3,1); \n' \
                 '\\fill[gray](3,0) rectangle (4,1); \n' \
                 '\\fill[gray](4,0) rectangle (5,1); \n' \
                 '\\fill[gray](0,1) rectangle (1,2); \n' \
                 '\\fill[gray](1,1) rectangle (2,2); \n' \
                 '\\fill[gray](2,1) rectangle (3,2); \n' \
                 '\\fill[gray](0,2) rectangle (1,3); \n' \
                 '\\draw[step=1cm,black,very thin] (0,0) grid (5,5); \n' \
                 '\\draw[step=1cm,blue,thick](0,4) rectangle (1,5); \n' \
                 '\\draw[step=1cm,green,thick](0,3) rectangle (1,4); \n' \
                 '\\draw[step=1cm,green,thick](1,2) rectangle (2,3); \n' \
                 '\\draw[step=1cm,green,thick](2,2) rectangle (3,3); \n' \
                 '\\draw[step=1cm,green,thick](3,1) rectangle (4,2); \n' \
                 '\\draw[step=1cm,green,thick](4,1) rectangle (5,2); \n' \
                 '\\end{tikzpicture}'

    print(boiler_start)
    print_heatmap(mdp)
    print(boiler_end)


def print_graph_latex_tikzpicture_CW(mdp):
    boiler_start = '\\begin{tikzpicture}  \n ' \
                   '\\draw[step=1cm,black,very thin] (0,0) grid (5,5); \n' \
                   '\\fill[blue](3,0) rectangle (4,1); \n' \
                   '\\fill[green](3,4) rectangle (4,5); \n' \
                   '\\fill[red] (4,4) rectangle (5,5); \n' \
                   '\\fill[red] (4,3) rectangle (5,5); \n' \
                   '\\fill[red] (4,2) rectangle (5,5); \n' \
                   '\\fill[red] (4,1) rectangle (5,5); \n' \
                   '\\fill[red] (4,0) rectangle (5,5); \n' \
                   '\n  \\foreach \\x/\\y/\\reward/\\action ' \
                   'in { '

    boiler_end = '} \n { \n' \
                 '\\ifthenelse{\\equal{\\action}{up} }{ \n' \
                 '\\draw [fill=orange,orange] (\\x - 0.35,\\y + 0.3) node[anchor=north]{} \n' \
                 '-- (\\x + 0.35 ,\\y + 0.3 ) node[anchor=north]{} \n' \
                 '-- (\\x,\\y + 0.45) node[anchor=south]{} \n' \
                 '-- cycle;}{} \n' \
                 '\\ifthenelse{\\equal{\\action}{down} }{ \n' \
                 '\\draw [fill=orange,orange] (\\x - 0.35,\\y - 0.3) node[anchor=north]{} \n' \
                 '-- (\\x + 0.35 ,\\y -0.3 ) node[anchor=north]{} \n' \
                 '-- (\\x,\\y - 0.45) node[anchor=south]{} \n' \
                 '-- cycle;}{} \n' \
                 '\\ifthenelse{\\equal{\\action}{left} }{ \n' \
                 '\\draw [fill=orange,orange] (\\x - 0.3,\\y - 0.35) node[anchor=north]{} \n' \
                 '-- (\\x - 0.3 ,\\y +0.35 ) node[anchor=north]{} \n' \
                 '-- (\\x - 0.45,\\y ) node[anchor=south]{} \n' \
                 '-- cycle;}{} \n' \
                 '\\ifthenelse{\\equal{\\action}{right} }{ \n' \
                 '\\draw [fill=orange,orange] (\\x + 0.3,\\y - 0.35) node[anchor=north]{} \n' \
                 '-- (\\x + 0.3 ,\\y + 0.35 ) node[anchor=north]{} \n' \
                 '-- (\\x + 0.45 ,\\y ) node[anchor=south]{} \n' \
                 '-- cycle;}{} \n' \
                 '\draw (\\x,\\y) node{\scriptsize\\reward}; \n' \
                 '} \n' \
                 '\\end{tikzpicture}'

    print(boiler_start)
    for s in mdp.get_states():
        if not s.get_solid():
            print_latex_for_state(s, mdp)
    print(boiler_end)


def print_graph_latex_tikzpicture_DST(mdp):

    boiler_start = '\\begin{tikzpicture}  \n \\fill[' \
                   'blue](3,0) rectangle (4,1); \n \\fill[blue](0,4) rectangle (1,5); \n' \
                   '\\fill[green](0,3) rectangle (1,4); \n' \
                   '\\fill[green](1,2) rectangle (2,3); \n' \
                   '\\fill[green](2,2) rectangle (3,3); \n' \
                   '\\fill[green](3,1) rectangle (4,2); \n' \
                   '\\fill[green](4,1) rectangle (5,2); \n' \
                   '\\fill[gray](0,0) rectangle (1,1); \n' \
                   '\\fill[gray](1,0) rectangle (2,1); \n' \
                   '\\fill[gray](2,0) rectangle (3,1); \n' \
                   '\\fill[gray](3,0) rectangle (4,1); \n' \
                   '\\fill[gray](4,0) rectangle (5,1); \n' \
                   '\\fill[gray](0,1) rectangle (1,2); \n' \
                   '\\fill[gray](1,1) rectangle (2,2); \n' \
                   '\\fill[gray](2,1) rectangle (3,2); \n' \
                   '\\fill[gray](0,2) rectangle (1,3); \n' \
                   '\\draw[step=1cm,black,very thin] (0,0) grid (5,5); \n' \
                   '\n  \\foreach \\x/\\y/\\reward/\\action ' \
                   'in { '

    boiler_end = '} \n { \n' \
                 '\\ifthenelse{\\equal{\\action}{up} }{ \n' \
                 '\\draw [fill=orange,orange] (\\x - 0.35,\\y + 0.3) node[anchor=north]{} \n' \
                 '-- (\\x + 0.35 ,\\y + 0.3 ) node[anchor=north]{} \n' \
                 '-- (\\x,\\y + 0.45) node[anchor=south]{} \n' \
                 '-- cycle;}{} \n' \
                 '\\ifthenelse{\\equal{\\action}{down} }{ \n' \
                 '\\draw [fill=orange,orange] (\\x - 0.35,\\y - 0.3) node[anchor=north]{} \n' \
                 '-- (\\x + 0.35 ,\\y -0.3 ) node[anchor=north]{} \n' \
                 '-- (\\x,\\y - 0.45) node[anchor=south]{} \n' \
                 '-- cycle;}{} \n' \
                 '\\ifthenelse{\\equal{\\action}{left} }{ \n' \
                 '\\draw [fill=orange,orange] (\\x - 0.3,\\y - 0.35) node[anchor=north]{} \n' \
                 '-- (\\x - 0.3 ,\\y +0.35 ) node[anchor=north]{} \n' \
                 '-- (\\x - 0.45,\\y ) node[anchor=south]{} \n' \
                 '-- cycle;}{} \n' \
                 '\\ifthenelse{\\equal{\\action}{right} }{ \n' \
                 '\\draw [fill=orange,orange] (\\x + 0.3,\\y - 0.35) node[anchor=north]{} \n' \
                 '-- (\\x + 0.3 ,\\y + 0.35 ) node[anchor=north]{} \n' \
                 '-- (\\x + 0.45 ,\\y ) node[anchor=south]{} \n' \
                 '-- cycle;}{} \n' \
                 '\draw (\\x,\\y) node{\scriptsize\\reward}; \n' \
                 '} \n' \
                 '\\end{tikzpicture}'

    print(boiler_start)
    for s in mdp.get_states():
        if not s.get_solid():
            print_latex_for_state(s, mdp)
    print(boiler_end)


def print_graph_latex_figure_CW(alpha):
    alpha_string = str(alpha)
    label_key = alpha_string[0: 1:] + alpha_string[2::]
    boiler_start = '\\begin{figure} \n \\centering'
    boiler_end = '\\caption{Optimal value functions $V_{LIMO}(\\alpha) = \\alpha V_{Cliff} + (1 - \\alpha)V_{' \
                 'Goal}$ (' \
                 'left) and $V_{MOMDP}(\\alpha)$ (right) for $\\alpha=' + alpha_string + '$ } \n ' \
                                                                                         '\\label{fig:CWCompareGraph' + label_key + '} \n' \
                                                                                                                                    '\\end{figure} '

    print(boiler_start)
    print_graph_latex_tikzpicture_CW(mdp3)
    print_graph_latex_tikzpicture_CW(mdp4)
    print(boiler_end)


def print_graph_latex_figure_DST(alpha):
    alpha_string = str(alpha)
    label_key = alpha_string[0: 1:] + alpha_string[2::]
    boiler_start = '\\begin{figure} \n \\centering'
    boiler_end = '\\caption{Optimal value functions $V_{LIMO}(\\alpha) = \\alpha V_{Time} + (1 - \\alpha)V_{' \
                 'Treasure}$ (' \
                 'left) and $V_{MOMDP}(\\alpha)$ (right) for $\\alpha=' + alpha_string + '$ } \n ' \
                                                                                         '\\label{fig:DSTCompareGraph' + label_key + '} \n' \
                                                                                                                                     '\\end{figure} '

    print(boiler_start)
    print_graph_latex_tikzpicture_DST(mdp3)
    print_graph_latex_tikzpicture_DST(mdp4)
    print(boiler_end)


def print_heatmap_latex_figure_CW(alpha):
    alpha_string = str(alpha)
    label_key = alpha_string[0: 1:] + alpha_string[2::]
    boiler_start = '\\begin{figure} \n \\centering'

    boiler_end = '\\caption{Heatmap of the states our agent visited during 1000 episodes for $\\alpha = ' + alpha_string + ' $ using LIMO (left) and a scarlized MOMDP policy (right) \n}' \
                                                                                                                           '\\label{fig:CWCompareHeatmap' + label_key + '} \n' \
                                                                                                                                                                        '\\end{figure} '

    print(boiler_start)
    print_heatmap_latex_tikzpicture_CW(mdp3)
    print_heatmap_latex_tikzpicture_CW(mdp4)
    print(boiler_end)


def print_heatmap_latex_figure_DST(alpha):
    alpha_string = str(alpha)
    label_key = alpha_string[0: 1:] + alpha_string[2::]
    boiler_start = '\\begin{figure} \n \\centering'

    boiler_end = '\\caption{Heatmap of the states our agent visited during 1000 episodes for $\\alpha = ' + alpha_string + ' $ using LIMO (left) and a scarlized MOMDP policy (right) \n}' \
                                                                                                                           '\\label{fig:DSTCompareHeatmap' + label_key + '} \n' \
                                                                                                                                                                         '\\end{figure} '

    print(boiler_start)
    print_heatmap_latex_tikzpicture_DST(mdp3)
    print_heatmap_latex_tikzpicture_DST(mdp4)
    print(boiler_end)


def print_full_appendix_comparisons():  # prints all the comparisons in latex-code for the selected environment f
    if config.LATEX_PRINT_MODE == 'single':
        print_full_latex_comparison(int(var_alpha.get())/100.0)
    else:
        for x in alpha_definition_set:
            print_full_latex_comparison(x)


def print_full_latex_comparison(alpha):
    environment = env_option.get()

    solve_mdps()
    run_episode(mdp3, 'LIMO', alpha)
    run_episode(mdp4, 'MO', alpha)

    if environment == 'cliff-world':
        print_heatmap_latex_figure_CW(alpha)
        print_graph_latex_figure_CW(alpha)
    else:
        print_heatmap_latex_figure_DST(alpha)
        print_graph_latex_figure_DST(alpha)


def print_chart_data():
    print('mdp1')
    for s in mdp1.get_states():
        if not s.get_solid():
            print_latex_for_state(s, mdp1)
    print('mdp2')
    for s in mdp2.get_states():
        if not s.get_solid():
            print_latex_for_state(s, mdp2)
    print('mdp3')
    for s in mdp3.get_states():
        if not s.get_solid():
            print_latex_for_state(s, mdp3)
    print('mdp4')
    for s in mdp4.get_states():
        if not s.get_solid():
            print_latex_for_state(s, mdp4)


def render_scrollbar_for_canvas(canvas_frame_input, canvas_input, col1, col2):
    scroll_x = Scrollbar(canvas_frame_input, orient="horizontal", command=canvas_input.xview)
    scroll_x.grid(row=1, column=col1, sticky="ew")
    scroll_y = Scrollbar(canvas_frame_input, orient="vertical", command=canvas_input.yview)
    scroll_y.grid(row=2, column=col2, sticky="ns")
    canvas_input.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    canvas_input.bind('<Configure>', lambda e: canvas_input.configure(scrollregion=canvas_input.bbox("all")))


def render_scrollbars():
    render_scrollbar_for_canvas(canvas_frame, c1, 0, 2)
    render_scrollbar_for_canvas(canvas_frame_2, c2, 1, 0)
    render_scrollbar_for_canvas(canvas_frame_3, c3, 2, 1)


if __name__ == '__main__':
    top = Tk()
    frame = Frame(top)
    frame.grid(row=0, column=0, sticky="w", padx=(10, 0))
    config = LIMOconfiguration()

    scale = 120 * 0.7
    xmdp1 = 0
    xmdp2 = 0
    padding = 25
    ymdp1 = 0
    ymdp3 = 0
    xmdp3 = 0

    var_size = StringVar(frame, value="5")
    var_number_of_episodes = StringVar(frame, value='1000')
    var_gamma = StringVar(frame, value='90')
    var_alpha = StringVar(frame, value='0')
    alpha_definition_set = create_definition_set(config.PREFERENCE_SET_SIZE)

    mdp1 = Mdp(var_size.get(), int(var_gamma.get()), [],config.NOISE_FACTOR)
    mdp2 = Mdp(var_size.get(), int(var_gamma.get()), [],config.NOISE_FACTOR)
    mdp3 = Mdp(var_size.get(), int(var_gamma.get()), [],config.NOISE_FACTOR)
    mdp4 = Mdp(var_size.get(), int(var_gamma.get()), [],config.NOISE_FACTOR)

    env_options = ['cliff-world', 'deep-sea-treasure']
    alg_options = ['LIMO', 'MOMPD']
    env_option = StringVar(value=env_options[0])
    alg_option = StringVar(value=alg_options[0])

    env_option.trace("w", lambda name, index, mode, sv=env_option: environment_change_event())
    var_alpha.trace("w", lambda name, index, mode, sv=var_alpha: set_lin_combination())
    var_gamma.trace("w", lambda name, index, mode, sv=var_gamma: environment_change_event())

    canvas_frame = Frame(top)
    Label(canvas_frame, text="LIMO Archetypes").grid(row=0, column=0, sticky="nw")
    canvas_frame.grid(row=2, column=0, sticky="w", padx=(10, 0))
    c1 = Canvas(canvas_frame, width=500, height=800, background='white')
    c1.grid(row=2, column=0, sticky="w")

    canvas_frame_2 = Frame(top)
    Label(canvas_frame_2, text="LIMO (top) and MOMDP (bottom) result").grid(row=0, column=1, sticky="nw")
    canvas_frame_2.grid(row=2, column=1, sticky="w")
    c2 = Canvas(canvas_frame_2, width=500, height=800, background='white')
    c2.grid(row=2, column=1, sticky="w")

    canvas_frame_3 = Frame(top)
    Label(canvas_frame_3, text="Heat Map").grid(row=0, column=2, sticky="nw")
    canvas_frame_3.grid(row=2, column=2, sticky="e")
    c3 = Canvas(canvas_frame_3, width=500, height=800, background='white')
    c3.grid(row=2, column=2, sticky="w")

    Label(frame, text="value").grid(row=1, column=1, sticky="nw")
    Label(frame, text="variable").grid(row=1, column=0, sticky="nw")
    Label(frame, text="alpha (%)").grid(row=2, column=0, sticky="w")
    Label(frame, text='Number of Episodes').grid(row=2, column=2, sticky='w')
    Label(frame, text="gamma (%)").grid(row=3, column=0, sticky="w")
    Label(frame, text="size (max 10)").grid(row=4, column=0, sticky="w")

    validation = (frame.register(validate_input))
    validation_size = (frame.register(validate_input_size))

    Entry(frame, validate='all', textvariable=var_alpha, validatecommand=(validation, '%P')).grid(row=2, column=1,
                                                                                                  sticky=E + W)
    Entry(frame, validate='all', textvariable=var_gamma, validatecommand=(validation, '%P')).grid(row=3, column=1,
                                                                                                  sticky=E + W)
    Entry(frame, validate='all', textvariable=var_number_of_episodes, validatecommand=(validation, '%P')).grid(
        row=2,
        column=3,
        sticky=E + W)
    Entry(frame, validate='all', textvariable=var_size, validatecommand=(validation_size, '%P')).grid(row=4,
                                                                                                      column=1,
                                                                                                      sticky=E + W)
    Button(frame, text="Increment Iteration",
           command=lambda: run_iteration(),
           activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=0, sticky="we")
    Button(frame, text="Solve Mdps",
           command=lambda: solve_mdps(),
           activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=1, sticky="we")
    Button(frame, text="resize Mdps",
           command=lambda: environment_change_event(),
           activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=2, sticky="we")

    Button(frame, text="Print Latex Comparison",
           command=lambda: print_full_appendix_comparisons(),
           activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=3, sticky="we")
    # Button(frame, text="Accu reward",
    #        command=lambda: accu_reward(c1, c2, mdp1, mdp2, mdp3, mdp4),
    #        activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=2, sticky="we")
    # Button(frame, text="run episode",
    #        command=lambda: episode_runner(c1, c2, mdp3, mdp4),
    #        activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=3, sticky="we")
    # Button(frame, text="print",
    #        command=lambda: print_chart_data(mdp1),
    #        activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=4, sticky="we")
    # Button(frame, text="single MO episode",
    #        command=lambda: run_single_episode(mdp4, 'mo'),
    #        activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=8, sticky="we")
    # Button(frame, text="single LIMO episode",
    #        command=lambda: run_single_episode(mdp3, 'limo'),
    #        activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=9, sticky="we")
    # Button(frame, text="draw graph limo",
    #        command=lambda: graph(mdp3, True, "LIMO"),
    #        activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=10, sticky="we")
    Button(frame, text="run (single preference)",
           command=lambda: run_single_preference(int(var_alpha.get()) / 100.0, False),
           activeforeground="red", activebackground="pink", pady=10).grid(row=4, column=2, sticky="we")
    Button(frame, text="run (multiple preferences)",
           command=lambda: run_multiple_preferences(),
           activeforeground="red", activebackground="pink", pady=10).grid(row=4, column=3, sticky="we")
    OptionMenu(frame, env_option, *env_options).grid(row=0, column=1)
    OptionMenu(frame, alg_option, *alg_options).grid(row=3, column=3)

    # These are the titles
    Label(frame, text='Select environment', width=15).grid(row=0, column=0)
    Label(frame, text='Approach', width=15).grid(row=3, column=2)

    start_cliff_world()
    render_scrollbars()
    top.mainloop()
