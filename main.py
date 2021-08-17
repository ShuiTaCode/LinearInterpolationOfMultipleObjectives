from tkinter import *
import numpy
import numpy as np
from matplotlib import pyplot as plt
from mdp import Mdp
import random


def start_cliff_world(m1, m2, m3):
    var_gamma_calc = int(var_gamma.get()) / 100
    m1.__init__(int(var_size.get()), var_gamma_calc, [])
    m2.__init__(int(var_size.get()), var_gamma_calc, [])
    m3.__init__(int(var_size.get()), var_gamma_calc, [])
    m1.set_environment('cliff-world')
    m2.set_environment('cliff-world')
    m3.set_environment('cliff-world')
    cliff = []
    start = {'x': int(var_size.get()) - 2, 'y': int(var_size.get()) - 1}
    end = {'x': int(var_size.get()) - 2, 'y': 0}

    m1.set_start(start)

    for i in range(int(var_size.get())):
        cliff.append({
            'x': int(var_size.get()) - 1,
            'y': i
        })
    m1.set_cliff(cliff
                 )

    m2.set_start(start)
    m2.set_end({'x': end['x'], 'y': end['y'], 'reward': 1})

    m3.set_start(start)
    m3.set_end({'x': end['x'], 'y': end['y'], 'reward': 1})
    m3.set_cliff(cliff)
    m3.increment_iteration()
    draw_graphs_and_policies(c1, c2, m1, m2, m3)


def get_solid_states_for_deep_sea(end_state_arr):
    res = []
    for state in end_state_arr:
        i = state.get_y() + 1
        while i < int(var_size.get()):
            res.append({'x': state.get_x(), 'y': i})
            i += 1
    return res


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
            'reward': x * y * 4
        })
    return result


def start_deep_sea_treasure(m1, m2, m3):
    var_gamma_calc = int(var_gamma.get()) / 100
    m1.__init__(int(var_size.get()), var_gamma_calc, [])
    m2.__init__(int(var_size.get()), var_gamma_calc, [])
    m3.__init__(int(var_size.get()), var_gamma_calc, [])
    m1.set_environment('cliff-world')
    m2.set_environment('cliff-world')
    m3.set_environment('cliff-world')
    m1.set_start({'x': 0, 'y': 0})
    m1.set_end({'x': int(var_size.get()) - 1, 'y': int(var_size.get()) - 1, 'reward': -1000})

    treasures = get_treasures_for_deep_sea()

    m2.set_start({'x': 0, 'y': 0})
    for treasure in treasures:
        m2.set_end(treasure)

    m3.set_start({'x': 0, 'y': 0})
    for treasure in treasures:
        m3.set_end(treasure)

    solidstates = get_solid_states_for_deep_sea([state for state in m2.get_states() if state.get_end()])
    mdp1.set_solid_states(solidstates)
    mdp2.set_solid_states(solidstates)
    mdp3.set_solid_states(solidstates)
    m3.increment_iteration()
    draw_graphs_and_policies(c1, c2, m1, m2, m3)
    print('new enviroment chosen')


def draw_square(c, x, y, w, h, color):
    c.create_line(x, y, x + w, y, fill=color, width=3)
    c.create_line(x + w, y, x + w, y + h, fill=color, width=3)
    c.create_line(x + w, y + h, x, y + h, fill=color, width=3)
    c.create_line(x, y + h, x, y, fill=color, width=3)


def draw_triangle(c, x, y, w, h, action, reward, color):
    stetch_width = 0.5  # stretch and compress in width of arrow
    delta = 30  # distance from the center of the square
    stretch_height = 0.5  # stretch and compress in height of arrow
    if reward < 999:
        if action == 'up':
            c.create_line(x, y - delta, x + stetch_width * w, y - delta, fill=color, width=3)
            c.create_line(x + stetch_width * w, y - delta, x, y - stetch_width * h * stretch_height - delta, fill=color,
                          width=3)
            c.create_line(x, y - stetch_width * h * stretch_height - delta, x - stetch_width * w, y - delta, fill=color,
                          width=3)
            c.create_line(x - stetch_width * w, y - delta, x, y - delta, fill=color, width=3)
            return
        if action == 'down':
            c.create_line(x, y + delta, x + stetch_width * w, y + delta, fill=color, width=3)
            c.create_line(x + stetch_width * w, y + delta, x, y + stetch_width * h * stretch_height + delta, fill=color,
                          width=3)
            c.create_line(x, y + stetch_width * h * stretch_height + delta, x - stetch_width * w, y + delta, fill=color,
                          width=3)
            c.create_line(x - stetch_width * w, y + delta, x, y + delta, fill=color, width=3)
            return
        if action == 'left':
            c.create_line(x - delta, y, x - delta, y - stetch_width * h, fill=color, width=3)
            c.create_line(x - delta, y - stetch_width * h, x - delta - stetch_width * w * stretch_height, y, fill=color,
                          width=3)
            c.create_line(x - stetch_width * w * stretch_height - delta, y, x - delta, y + stetch_width * h, fill=color,
                          width=3)
            c.create_line(x - delta, y + stetch_width * h, x - delta, y, fill=color, width=3)
            return
        if action == 'right':
            c.create_line(x + delta, y, x + delta, y - stetch_width * h, fill=color, width=3)
            c.create_line(x + delta, y - stetch_width * h, x + stetch_width * w * stretch_height + delta, y, fill=color,
                          width=3)
            c.create_line(x + stetch_width * w * stretch_height + delta, y, x + delta, y + stetch_width * h, fill=color,
                          width=3)
            c.create_line(x + delta, y + stetch_width * h, x + delta, y, fill=color, width=3)
            return


def validate_input(P):
    if (P.isdigit() and int(P) <= 100) or P == "":
        return True
    else:
        return False


def draw_graph(c, x, y, scale, mdp):
    temp_init_state = {}
    end_states = []
    solid_states = []
    for s in mdp.get_states():
        if s.get_solid():
            solid_states.append(s)
        elif s.get_start():
            temp_init_state = s
        elif s.get_end() or mdp.part_of_cliff(s):
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


def draw_policy(c, x, y, scale, set_of_states):
    for s in set_of_states:
        if len(s.get_add_values()) < 4:
            for max_value in s.get_add_values():
                draw_triangle(c, x + s.x * scale + 0.5 * scale, y + s.y * scale + 0.5 * scale, 0.5 * scale, 0.5 * scale,
                              max_value['a'], s.value['r'], 'grey')


def run_iteration(c1, c2, m1, m2, m3):
    m1.increment_iteration()
    m2.increment_iteration()
    draw_graphs_and_policies(c1, c2, m1, m2, m3)


def calc_lin_combination(mdp1, mdp2, mdp3, alpha):
    mdp1_states = mdp1.get_states()
    mdp2_states = mdp2.get_states()
    mdp3_states = mdp3.get_states()
    new_states = []

    for i in range(len(mdp3_states)):
        new_state = mdp3_states[i]
        new_value = {'a': new_state.get_value()['a'],
                     'r': mdp1_states[i].get_value()['r'] * alpha + mdp2_states[i].get_value()['r'] * (1 - alpha)}
        # if not new_state.get_end() and not mdp3.part_of_cliff(new_state):
        new_state.set_value(new_value)

        # if mdp1_states[i].get_end() or mdp2_states[i].get_end():
        #   new_state.set_end(True)
        # else:
        #   new_state.set_end(False)
        new_states.append(new_state)

    mdp3.set_states(new_states)
    mdp3.eval_policy()
    # mdp3.run_iteration()
    print('combination finished')
    return mdp3.get_states()


def solve_mds(c1, c2, m1, m2, m3):
    mdp1.solve_mdp(var_size)
    mdp2.solve_mdp(var_size)
    draw_graphs_and_policies(c1, c2, m1, m2, m3)


def draw_graphs_and_policies(canvas1, canvas2, mdp1n, mdp2n, mdp3n):
    ymdp2 = int(var_size.get()) * scale * 0.7 + padding
    c1.delete('all')
    c2.delete('all')
    draw_policy(canvas1, xmdp1, ymdp1, scale * 0.7, mdp1n.get_states())
    draw_policy(canvas1, xmdp2, ymdp2, scale * 0.7, mdp2n.get_states())
    draw_policy(canvas2, xmdp3, ymdp3, scale * 0.7, mdp3n.get_states())
    draw_graph(canvas1, xmdp1, ymdp1, scale * 0.7, mdp1n)
    draw_graph(canvas1, xmdp2, ymdp2, scale * 0.7, mdp2n)
    draw_graph(canvas2, xmdp3, ymdp3, scale * 0.7, mdp3n)
    render_scrollbars(canvas1, canvas2)


def run_multi_safe_alg(c1, c2, input, m1, m2, m3):
    m1.solve_mdp(var_size)
    m2.solve_mdp(var_size)
    m3.set_states(calc_lin_combination(m1, m2, m3, input / 100))
    # mdp3.eval_policy()
    draw_graphs_and_policies(c1, c2, m1, m2, m3)


def accu_reward(c1, c2, m1, m2, m3):
    mdp1_data = []
    mdp2_data = []
    sum_data = []
    episode_data = []
    mdp1.solve_mdp(var_size)
    mdp2.solve_mdp(var_size)
    c1.delete('all')
    c2.delete('all')
    alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1.00]
    for x in alpha:
        m3.set_states(calc_lin_combination(mdp1, mdp2, mdp3, x))
        # mdp3.run_iteration()
        # m3.eval_policy()
        #    if x>0 and x<100:
        episode_data.append(run_episode(c1, c2))

        reward_mdp1 = m1.return_start().get_value()['r'] * x
        reward_mdp2 = m2.return_start().get_value()['r'] * (1 - x)
        sum = reward_mdp1 + reward_mdp2

        mdp1_data.append(reward_mdp1)
        mdp2_data.append(reward_mdp2)
        sum_data.append(sum)
        # print('final reward mdp1 for alpha ' + str(x), reward_mdp1)
        # print('final reward mdp2 for alpha ' + str(x), reward_mdp2)

    draw_graphs_and_policies(c1, c2, m1, m2, m3)

    # mdp3.print_prob()
    fig = plt.figure()
    plot_graph(mdp1_data, mdp2_data, sum_data, episode_data, fig)
    plot_episode_count(episode_data, fig)
    plot_episode_graph(episode_data, fig)
    plt.show()


def run_episode(c1, c2):
    pos_count = 0
    neg_count = 0
    pos_reward = []
    neg_reward = []
    count = []
    for i in range(50):
        res = mdp3.run_episode()
        if res['success']:
            pos_count += 1
            count.append(res['iteration'])
            pos_reward.append(res['discounted_reward'])
        else:
            neg_count += 1
            count.append(res['iteration'])
            neg_reward.append(res['discounted_reward'])
    # print('this is 1000 count', count)
    mean_pos = np.mean(pos_reward)
    median_pos = np.median(pos_reward)
    mean_neg = np.mean(neg_reward)
    median_neg = np.median(neg_reward)
    print('mean and median pos', np.mean(pos_reward), np.median(pos_reward))
    return {
        'pos': pos_count,
        'neg': neg_count,
        'count': count,
        'pos_reward': pos_reward,
        'neg_reward': neg_reward,
    }


def plot_episode_count(episode_data, fig):
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1.00]
    count = []
    for episode in episode_data:
        count.append(numpy.median(episode['count']))

    ax1 = fig.add_subplot(223)

    ax1.scatter(x, count, s=10, c='b', marker="s", label='number of transitions')
    plt.legend(loc='upper left')


def plot_episode_graph(episode_data, fig):
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1.00]
    pos = []
    neg = []
    for episode in episode_data:
        pos.append(episode['pos'])
        neg.append(episode['neg'])

    ax1 = fig.add_subplot(222)

    ax1.scatter(x, pos, s=10, c='b', marker="s", label='pos')
    ax1.scatter(x, neg, s=10, c='r', marker="o", label='neg')
    plt.legend(loc='upper left')


def plot_graph(mdp1_data, mdp2_data, sum, episode_data, fig):
    # print('what is episode_data', episode_data)

    pos_data = []
    neg_data = []
    for episode in episode_data:
        pos_data.append(numpy.mean(episode['pos_reward']))
        neg_data.append(numpy.mean(episode['neg_reward']))

    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1.00]
    # print('size',len(mdp1_data),len(mdp2_data),len(x))
    # fig = plt.figure()
    ax1 = fig.add_subplot(221)

    ax1.scatter(x, mdp2_data, s=10, c='b', marker="o", label='right Mdp')
    ax1.scatter(x, pos_data, s=10, c='#add8e6', marker="s", label='pos_data Mdp')
    ax1.scatter(x, mdp1_data, s=10, c='r', marker="o", label='left Mdp')
    ax1.scatter(x, neg_data, s=10, c='#f7a6a8', marker="s", label='neg_data Mdp')
    # ax1.scatter(x, sum, s=10, c='g', marker="o", label='both')
    plt.legend(loc='upper right')


def environment_change_event(m1, m2, m3):
    print('environment changed: ', env_option.get())
    option = env_option.get()
    if option == 'cliff-world':
        start_cliff_world(m1, m2, m3)
    if option == 'deep-sea-treasure':
        start_deep_sea_treasure(m1, m2, m3)


def render_scrollbars(canvas1, canvas2):
    # canvas_frame.forget()
    scroll_x = Scrollbar(canvas_frame, orient="horizontal", command=canvas1.xview)
    scroll_x.grid(row=2, column=0, sticky="ew")
    scroll_y = Scrollbar(canvas_frame, orient="vertical", command=canvas1.yview)
    scroll_y.grid(row=1, column=2, sticky="ns")

    canvas1.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    canvas1.bind('<Configure>', lambda e: canvas1.configure(scrollregion=canvas1.bbox("all")))
    # canvas_frame_2.forget()
    scroll_x_2 = Scrollbar(canvas_frame_2, orient="horizontal", command=canvas2.xview)
    scroll_x_2.grid(row=2, column=1, sticky="ew")
    scroll_y_2 = Scrollbar(canvas_frame_2, orient="vertical", command=canvas2.yview)
    scroll_y_2.grid(row=1, column=2, sticky="ns")

    canvas2.configure(yscrollcommand=scroll_y_2.set, xscrollcommand=scroll_x_2.set)
    canvas2.bind('<Configure>', lambda e: canvas2.configure(scrollregion=canvas2.bbox("all")))


if __name__ == '__main__':
    top = Tk()
    frame = Frame(top)
    frame.grid(row=0, column=0, sticky="w", padx=(10, 0))

    scale = 120
    # size = 5
    var_size = StringVar(frame, value=5)
    xmdp1 = 0
    xmdp2 = 0
    padding = 25
    ymdp1 = 0
    ymdp3 = 0
    xmdp3 = 0
    scale_mdp_3 = 500 / int(var_size.get())

    gamma = 0.9
    mdp1 = Mdp(var_size.get(), gamma, [])
    mdp2 = Mdp(var_size.get(), gamma, [])
    mdp3 = Mdp(var_size.get(), gamma, [])
    var_alpha = StringVar(frame, value='0')
    var_alpha.trace("w", lambda name, index, mode, sv=var_alpha: run_multi_safe_alg(c1, c2, int(var_alpha.get()), mdp1,
                                                                                    mdp2, mdp3))
    var_gamma = StringVar(frame, value='90')
    var_gamma.trace("w",
                    lambda name, index, mode, sv=var_gamma: environment_change_event(mdp1, mdp2, mdp3))

    # stvar = StringVar()
    # stvar.set("one")

    env_options = ['cliff-world', 'deep-sea-treasure']
    env_option = StringVar(value=env_options[0])
    # env_option.set(env_options[0])  # the first value
    env_option.trace("w", lambda name, index, mode, sv=env_option: environment_change_event(mdp1, mdp2, mdp3))

    canvas_frame = Frame(top)
    canvas_frame.grid(row=1, column=0, padx=(10, 0))
    c1 = Canvas(canvas_frame, width=700, height=500, background='white')
    c1.grid(row=1, column=0)

    canvas_frame_2 = Frame(top)
    canvas_frame_2.grid(row=1, column=1)
    c2 = Canvas(canvas_frame_2, width=700, height=500, background='white')
    c2.grid(row=1, column=1)

    render_scrollbars(c1, c2)

    label_var = Label(frame, text="value").grid(row=1, column=1, sticky="nw")
    label_val = Label(frame, text="variable").grid(row=1, column=0, sticky="nw")

    label_alpha = Label(frame, text="alpha").grid(row=2, column=0, sticky="w")
    label_gamma = Label(frame, text="gamma").grid(row=3, column=0, sticky="w")
    label_size = Label(frame, text="size").grid(row=4, column=0, sticky="w")

    validation = (frame.register(validate_input))

    e1 = Entry(frame, validate='all', textvariable=var_alpha, validatecommand=(validation, '%P')).grid(row=2, column=1,
                                                                                                       sticky=E + W)

    e2 = Entry(frame, validate='all', textvariable=var_gamma, validatecommand=(validation, '%P')).grid(row=3, column=1,
                                                                                                       sticky=E + W)

    e3 = Entry(frame, validate='all', textvariable=var_size, validatecommand=(validation, '%P')).grid(row=4,
                                                                                                      column=1,
                                                                                                      sticky=E + W)
    be3 = Button(frame, text="resize Mdp's",
                 command=lambda: environment_change_event(mdp1, mdp2, mdp3),
                 activeforeground="red", activebackground="pink", pady=10).grid(row=4, column=2, sticky="we")

    b1 = Button(frame, text="Solve Mdps",
                command=lambda: solve_mds(c1, c2, mdp1, mdp2, mdp3),
                activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=0, sticky="we")
    b2 = Button(frame, text="Linear Combination",
                command=lambda: run_multi_safe_alg(c1, c2, int(var_alpha.get()), mdp1, mdp2, mdp3),
                activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=1, sticky="we")
    b3 = Button(frame, text="Increment Iteration",
                command=lambda: run_iteration(c1, c2, mdp1, mdp2, mdp3),
                activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=4, sticky="we")
    b4 = Button(frame, text="Accu reward",
                command=lambda: accu_reward(c1, c2, mdp1, mdp2, mdp3),
                activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=2, sticky="we")
    b5 = Button(frame, text="run episode",
                command=lambda: run_episode(c1, c2),
                activeforeground="red", activebackground="pink", pady=10).grid(row=5, column=3, sticky="we")
    w = OptionMenu(frame, env_option, *env_options).grid(row=0, column=1)

    # These are the titles
    l1 = Label(frame, text='Select environment', width=15)
    l1.grid(row=0, column=0)
    start_cliff_world(mdp1, mdp2, mdp3)
    run_iteration(c1, c2, mdp1, mdp2, mdp3)
    # draw_graphs_and_policies(c)
    top.mainloop()
