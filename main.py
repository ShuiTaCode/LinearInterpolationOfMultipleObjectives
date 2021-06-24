from pandas import DataFrame
from tkinter import *

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# initialStateDistribution = [0.1, 0.05, 0.05, 0.2, 0.4, 0.2]
# initialStateId = np.random.choice(init_set_of_states).id
from mdp import Mdp
from state import State

canvas_width = 1460
scale = 100
size = 5
xmdp1 = 0
padding = 25
mdp_width = size * scale
x_end_mdp1 = xmdp1 + mdp_width * 0.7
xmdp2 = canvas_width - (mdp_width * 0.7)
space = xmdp2 - x_end_mdp1 - 2 * padding
xmdp3 = x_end_mdp1 + padding
scale_mdp_3 = space / size

gamma = 0.9
mdp1 = Mdp(size, gamma, [], False)
mdp1.set_start({'x': 1, 'y': 4})
mdp1.set_end({'x': 3, 'y': 0})
mdp1.set_cliff([
    {'x': 4, 'y': 0},
    {'x': 4, 'y': 1},
    {'x': 4, 'y': 2},
    {'x': 4, 'y': 3},
    {'x': 4, 'y': 4}
]
)
# policy1 = mdp1.solve_mdp()
mdp2 = Mdp(size, gamma, [], True)
mdp2.set_start({'x': 1, 'y': 4})
mdp2.set_end({'x': 3, 'y': 0})
mdp2.set_cliff([
    {'x': 4, 'y': 0},
    {'x': 4, 'y': 1},
    {'x': 4, 'y': 2},
    {'x': 4, 'y': 3},
    {'x': 4, 'y': 4}
])

mdp3 = Mdp(size, gamma, [], True)
mdp3.set_start({'x': 1, 'y': 4})
mdp3.set_end({'x': 3, 'y': 0})
mdp3.set_cliff([
    {'x': 4, 'y': 0},
    {'x': 4, 'y': 1},
    {'x': 4, 'y': 2},
    {'x': 4, 'y': 3},
    {'x': 4, 'y': 4}
])
# policy2 = mdp2.solve_mdp()

coef_mdp_left = 0
coef_mdp_right = 0


def draw_square(c, x, y, w, h, color):
    c.create_line(x, y, x + w, y, fill=color, width=3)
    c.create_line(x + w, y, x + w, y + h, fill=color, width=3)
    c.create_line(x + w, y + h, x, y + h, fill=color, width=3)
    c.create_line(x, y + h, x, y, fill=color, width=3)


def draw_rectangle(c, x, y, w, h, action, reward, color):
    if reward < 999:
        if action == 'up':
            c.create_line(x, y, x + 0.5 * w, y, fill=color, width=3)
            c.create_line(x + 0.5 * w, y, x, y - 0.5 * h, fill=color, width=3)
            c.create_line(x, y - 0.5 * h, x - 0.5 * w, y, fill=color, width=3)
            c.create_line(x - 0.5 * w, y, x, y, fill=color, width=3)
            return
        if action == 'down':
            c.create_line(x, y, x + 0.5 * w, y, fill=color, width=3)
            c.create_line(x + 0.5 * w, y, x, y + 0.5 * h, fill=color, width=3)
            c.create_line(x, y + 0.5 * h, x - 0.5 * w, y, fill=color, width=3)
            c.create_line(x - 0.5 * w, y, x, y, fill=color, width=3)
            return
        if action == 'left':
            c.create_line(x, y, x, y - 0.5 * h, fill=color, width=3)
            c.create_line(x, y - 0.5 * h, x - 0.5 * w, y, fill=color, width=3)
            c.create_line(x - 0.5 * w, y, x, y + 0.5 * h, fill=color, width=3)
            c.create_line(x, y + 0.5 * h, x, y, fill=color, width=3)
            return
        if action == 'right':
            c.create_line(x, y, x, y - 0.5 * h, fill=color, width=3)
            c.create_line(x, y - 0.5 * h, x + 0.5 * w, y, fill=color, width=3)
            c.create_line(x + 0.5 * w, y, x, y + 0.5 * h, fill=color, width=3)
            c.create_line(x, y + 0.5 * h, x, y, fill=color, width=3)
            return


def validate_input(P):
    if (P.isdigit() and int(P) <= 100) or P == "":
        return True
    else:
        return False


def draw_graph(c, x, y, scale, set_of_states):
    temp_init_state = {}
    end_states = []
    for s in set_of_states:
        if s.get_start():
            temp_init_state = s
        elif s.get_end():
            end_states.append(s)
        else:
            draw_square(c, x + s.x * scale, y + s.y * scale, scale, scale, 'black')
            c.create_text(x + s.x * scale + scale / 2, y + s.y * scale + scale / 2, text=s.get_value()['r'],
                          anchor='nw',
                          font='TkMenuFont', fill='black')
    draw_square(c, x + temp_init_state.get_x() * scale, y + temp_init_state.get_y() * scale, scale, scale, 'blue')
    c.create_text(x + temp_init_state.x * scale, y + temp_init_state.y * scale, text='Start', anchor='nw',
                  font='TkMenuFont', fill='blue')
    c.create_text(x + temp_init_state.x * scale + scale / 2, y + temp_init_state.y * scale + scale / 2,
                  text=temp_init_state.get_value()['r'], anchor='nw',
                  font='TkMenuFont', fill='black')

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

        c.create_text(x + temp_end_state.x * scale + scale / 2, y + temp_end_state.y * scale + scale / 2,
                      text=temp_end_state.get_value()['r'], anchor='nw',
                      font='TkMenuFont', fill='black')


def draw_policy(c, x, y, scale, set_of_states):
    for s in set_of_states:
        for max_value in s.get_add_values():
            draw_rectangle(c, x + s.x * scale + 0.5 * scale, y + s.y * scale + 0.5 * scale, 0.5 * scale, 0.5 * scale,
                           max_value['a'], s.value['r'], 'pink')


def draw_policy2(c, x, y, scale, set_of_states):
    for s in set_of_states:
        for max_value in s.get_add_values():
            draw_rectangle(c, x + s.x * scale + 0.5 * scale, y + s.y * scale + 0.5 * scale, 0.5 * scale, 0.5 * scale,
                           max_value['a'], s.value['r'], 'pink')


def run_iteration(c):
    mdp1.run_iteration()
    mdp2.run_iteration()
    # mdp3.set_states(calc_lin_combination(mdp1.get_states(), mdp2.get_states(), mdp3.get_states(), int(v1.get()) / 100,
    #                                     int(v2.get()) / 100))
    c.delete('all')
    draw_policy2(c, xmdp1, 0, scale * 0.7, mdp1.get_states())
    draw_policy(c, xmdp2, 0, scale * 0.7, mdp2.get_states())
    #draw_policy(c, xmdp3, 0, scale_mdp_3, mdp3.get_states())
    draw_graph(c, xmdp1, 0, scale * 0.7, mdp1.get_states())
    draw_graph(c, xmdp2, 0, scale * 0.7, mdp2.get_states())
    draw_graph(c, xmdp3, 0, scale_mdp_3, mdp3.get_states())


def run_safe_iteration(c):
    mdp2.run_safe_iteration()
    c.delete('all')
    draw_policy(c, xmdp1, 100, scale * 0.7, mdp1.get_states())
    draw_policy(c, canvas_width - size * scale * 0.7, 100, scale * 0.7, mdp2.get_states())
    draw_graph(c, xmdp1, 100, scale * 0.7, mdp1.get_states())
    draw_graph(c, canvas_width - size * scale * 0.7, 100, scale * 0.7, mdp2.get_states())


def calc_lin_combination(mdp1_states, mdp2_states, mdp3_states, alpha):
    new_states = []

    for i in range(len(mdp3_states)):
        new_state = mdp3_states[i]
        new_value = {'a': new_state.get_value()['a'],
                     'r': mdp1_states[i].get_value()['r'] * alpha + mdp2_states[i].get_value()['r'] * (1 - alpha)}
        new_state.set_value(new_value)

        if mdp1_states[i].get_end() or mdp2_states[i].get_end():
            new_state.set_end(True)
        else:
            new_state.set_end(False)
        new_states.append(new_state)

    mdp3.set_states(new_states)
    #    mdp3.eval_policy()
    # mdp3.run_iteration()

    return mdp3.get_states()


def solve_mds(c):
    mdp1.solve_mdp()
    mdp2.solve_mdp()
    c.delete('all')
    draw_policy(c, xmdp1, 0, scale * 0.7, mdp1.get_states())
    draw_policy(c, xmdp2, 0, scale * 0.7, mdp2.get_states())
    draw_policy(c, xmdp3, 0, scale_mdp_3, mdp3.get_states())
    draw_graph(c, xmdp1, 0, scale * 0.7, mdp1.get_states())
    draw_graph(c, xmdp2, 0, scale * 0.7, mdp2.get_states())
    draw_graph(c, xmdp3, 0, scale_mdp_3, mdp3.get_states())


def run_multi_safe_alg(c,input):
    mdp1.solve_mdp()
    mdp2.solve_mdp()
    mdp3.set_states(calc_lin_combination(mdp1.get_states(), mdp2.get_states(), mdp3.get_states(), input / 100))
    # mdp3.run_iteration()
    mdp3.eval_policy()
    c.delete('all')

    draw_policy(c, xmdp1, 0, scale * 0.7, mdp1.get_states())
    draw_policy(c, xmdp2, 0, scale * 0.7, mdp2.get_states())
    draw_policy(c, xmdp3, 0, scale_mdp_3, mdp3.get_states())
    draw_graph(c, xmdp1, 0, scale * 0.7, mdp1.get_states())
    draw_graph(c, xmdp2, 0, scale * 0.7, mdp2.get_states())
    draw_graph(c, xmdp3, 0, scale_mdp_3, mdp3.get_states())

def test_reward():
    states_mdp3 = mdp3.accu_states([], {}, mdp3.return_start())

    reward_mdp1 = 0
    reward_mdp2 = 0
    for state in states_mdp3:
            print(state.get_x(),state.get_y())
            for state_mdp1 in mdp1.get_states():
                if state_mdp1.get_x()==state.get_x() and state_mdp1.get_y()==state.get_y():
                    reward_mdp1 += state_mdp1.get_value()['r']
            for state_mdp2 in mdp2.get_states():
                if state_mdp2.get_x()==state.get_x() and state_mdp2.get_y()==state.get_y():
                    reward_mdp2 += state_mdp2.get_value()['r']

    print('final reward mdp1 for alpha ' + str(int(v1.get()) / 100),reward_mdp1)
    print('final reward mdp2 for alpha ' + str(int(v1.get()) / 100),reward_mdp2)

def accu_reward():
    mdp1_data=[]
    mdp2_data=[]

    for x in range(0,100,10):
        run_multi_safe_alg(c,x)
        states_mdp3 = mdp3.accu_states([],{},mdp3.return_start())

        reward_mdp1 = 0
        reward_mdp2 = 0

        print('ACCU')
        for state in states_mdp3:
            print(state.get_x(),state.get_y())
            for state_mdp1 in mdp1.get_states():
                if state_mdp1.get_x()==state.get_x() and state_mdp1.get_y()==state.get_y():
                    reward_mdp1 += state_mdp1.get_value()['r']
            for state_mdp2 in mdp2.get_states():
                if state_mdp2.get_x()==state.get_x() and state_mdp2.get_y()==state.get_y():
                    reward_mdp2 += state_mdp2.get_value()['r']
        mdp1_data.append(reward_mdp1)
        mdp2_data.append(reward_mdp2)
        print('final reward mdp1 for alpha ' + str(x),reward_mdp1)
        print('final reward mdp2 for alpha ' + str(x),reward_mdp2)
    plot_graph(mdp1_data,mdp2_data)


def plot_graph(mdp1_data,mdp2_data):
    x = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    print('size',len(mdp1_data),len(mdp2_data),len(x))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x, mdp1_data, s=10, c='b', marker="s", label='first')
    ax1.scatter(x, mdp2_data, s=10, c='r', marker="o", label='second')
    plt.legend(loc='upper right')
    plt.show()



# mdp3.set_states(calc_lin_combination(mdp1.get_states(), mdp2.get_states(), mdp3.get_states(), 0.5, 0.5))






if __name__ == '__main__':
    top = Tk()
    # top.geometry("1260x960")
    # creating a simple canvas

    # self.root = root
    # self.entry = tk.Entry(root)
    stvar = StringVar()
    stvar.set("one")

    c = Canvas(top, width=canvas_width, height=1000, background='white')
    c.grid(row=1, column=0)

    frame = Frame(top)
    frame.grid(row=0, column=0, sticky="n")

    # option = OptionMenu(frame, stvar, "one", "two", "three")
    label1 = Label(frame, text="coefficient").grid(row=0, column=1, sticky="nw")
    label2 = Label(frame, text="alpha (Mdp left)").grid(row=1, column=0, sticky="w")
    label3 = Label(frame, text="beta (Mdp right)").grid(row=2, column=0, sticky="w")
    # option.grid(row=0, column=1, sticky="nwe")
    # entry = Entry(frame).grid(row=1, column=1, sticky=E + W)
    validation = (frame.register(validate_input))
    v1 = StringVar(frame, value='0')



    e1 = Entry(frame, validate='all', textvariable=v1, validatecommand=(validation, '%P')).grid(row=1, column=1,
                                                                                                sticky=E + W)



    # coef_mdp_left=e1.get()
    # coef_mdp_right=e2.get()
    # Button1 = Button(frame, text="Draw").grid(row=3, column=1, sticky="we")
    # figure1 = c.create_rectangle(80, 80, 120, 120, fill="blue")

    b1 = Button(frame, text="Solve Mdps",
                command=lambda: solve_mds(c),
                activeforeground="red", activebackground="pink", pady=10).grid(row=3, column=0, sticky="we")
    b2 = Button(frame, text="Linear Combination",
                command=lambda: run_multi_safe_alg(c,int(v1.get())),
                activeforeground="red", activebackground="pink", pady=10).grid(row=3, column=1, sticky="we")
    b3 = Button(frame, text="Increment Iteration",
                command=lambda: run_iteration(c),
                activeforeground="red", activebackground="pink", pady=10).grid(row=3, column=2, sticky="we")
    b4 = Button(frame, text="Accu reward",
                command=lambda: accu_reward(),
                activeforeground="red", activebackground="pink", pady=10).grid(row=3, column=3, sticky="we")

    #    b1.pack(side=TOP)
    # draw_policy(c, xmdp1, 0, scale * 0.7, mdp1.get_states())
    # draw_policy(c, xmdp2, 0, scale * 0.7, mdp2.get_states())
    draw_graph(c, xmdp1, 0, scale * 0.7, mdp1.get_states())
    draw_graph(c, xmdp2, 0, scale * 0.7, mdp2.get_states())
    # draw_graph(c, xmdp3, 0, scale_mdp_3, mdp3.get_states())

    #    c.pack()






    top.mainloop()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
