from tkinter import *

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

gamma = 1
mdp1 = Mdp(size, gamma, [])
policy1 = mdp1.solve_mdp()
mdp2 = Mdp(size, gamma, [])
policy2 = mdp2.solve_mdp()





def draw_square(c, x, y, w, h, color):
    c.create_line(x, y, x + w, y, fill=color, width=3)
    c.create_line(x + w, y, x + w, y + h, fill=color, width=3)
    c.create_line(x + w, y + h, x, y + h, fill=color, width=3)
    c.create_line(x, y + h, x, y, fill=color, width=3)


def draw_rectangle(c, x, y, w, h, action, reward, color):
    if reward > 0:
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

    # draw_square(c, x + temp_init_state.x * scale, y + temp_init_state.y * scale, scale, scale, 'blue')
    # c.create_text(x + temp_init_state.x * scale, y + temp_init_state.y * scale, text='Start', anchor='nw',
    #             font='TkMenuFont', fill='blue')
    # c.create_text(x + temp_init_state.x * scale + scale / 2, y + temp_init_state.y * scale + scale / 2,
    #            text=temp_init_state.get_value()['r'], anchor='nw',
    #           font='TkMenuFont', fill='black')
    for temp_end_state in end_states:
        draw_square(c, x + temp_end_state.x * scale, y + temp_end_state.y * scale, scale, scale, 'green')

        c.create_text(x + temp_end_state.x * scale, y + temp_end_state.y * scale, text='End ', anchor='nw',
                      font='TkMenuFont', fill='green')
        c.create_text(x + temp_end_state.x * scale + scale / 2, y + temp_end_state.y * scale + scale / 2,
                      text=temp_end_state.get_value()['r'], anchor='nw',
                      font='TkMenuFont', fill='black')


def draw_policy(c, x, y, scale, set_of_states):
    for s in set_of_states:
        for max_value in s.get_add_values():
            draw_rectangle(c, x + s.x * scale + 0.5 * scale, y + s.y * scale + 0.5 * scale, 0.5 * scale, 0.5 * scale,
                           max_value['a'], s.value['r'], 'pink')


def run_iteration(c):
    print('iteration wird ausgeführt_main')

    mdp1.run_iteration()
    mdp2.run_iteration()
    c.delete('all')
    draw_policy(c, xmdp1, 100, scale * 0.7, mdp1.get_states())
    draw_policy(c, canvas_width - size * scale * 0.7, 100, scale * 0.7, mdp2.get_states())
    draw_graph(c, xmdp1, 100, scale * 0.7, mdp1.get_states())
    draw_graph(c, canvas_width - size * scale * 0.7, 100, scale * 0.7, mdp2.get_states())


def run_safe_iteration(c):
    mdp2.run_safe_iteration()
    c.delete('all')
    draw_policy(c, xmdp1, 100, scale * 0.7, mdp1.get_states())
    draw_policy(c, canvas_width - size * scale * 0.7, 100, scale * 0.7, mdp2.get_states())
    draw_graph(c, xmdp1, 100, scale * 0.7, mdp1.get_states())
    draw_graph(c, canvas_width - size * scale * 0.7, 100, scale * 0.7, mdp2.get_states())



def calc_lin_combination(mdp1_states, mdp2_states, mdp3_states, coef_mdp1, coef_mdp2):
    new_states = []

    for i in range(len(mdp3_states)):
        new_state = mdp3_states[i]
        new_value = {'a': new_state.get_value()['a'],
                     'r': mdp1_states[i].get_value()['r'] * coef_mdp1 + mdp2_states[i].get_value()['r'] * coef_mdp2}
        new_state.set_value(new_value)
        print('new value',new_value['r'],mdp1_states[i].get_value()['r'],mdp2_states[i].get_value()['r'])
        if(mdp1_states[i].get_end() or mdp2_states[i].get_end()):
            new_state.set_end(True)
        else:
            new_state.set_end(False)
        new_states.append(new_state)


    mdp3.set_states(new_states)
    mdp3.eval_policy()
    #mdp3.run_iteration()

    return mdp3.get_states()

def run_multi_safe_alg(c):
    mdp3.set_states(calc_lin_combination(mdp1.get_states(),mdp2.get_states(),mdp3.get_states(),0.5,0.5))
    #mdp3.run_iteration()
    mdp3.eval_policy()
    c.delete('all')
    print('mdp3 states')
    for state in mdp3.get_states():
        print(state.__dict__)
    draw_policy(c, xmdp1, 100, scale * 0.7, mdp1.get_states())
    draw_policy(c, xmdp2, 100, scale * 0.7, mdp2.get_states())
    draw_policy(c, xmdp3, 100, scale_mdp_3 , mdp3.get_states())
    draw_graph(c, xmdp1, 100, scale * 0.7, mdp1.get_states())
    draw_graph(c, xmdp2, 100, scale * 0.7, mdp2.get_states())
    draw_graph(c, xmdp3, 100, scale_mdp_3, mdp3.get_states())

mdp3 = Mdp(size, gamma, [])
mdp3.set_states(calc_lin_combination(mdp1.get_states(),mdp2.get_states(),mdp3.get_states(),0.5,0.5))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    top = Tk()
    top.geometry("1260x960")
    # creating a simple canvas
    c = Canvas(top, bg="white", height="960", width="1460")
    b1 = Button(top, text="Increment Iteration",
                command=lambda: run_multi_safe_alg(c),
                activeforeground="red", activebackground="pink", pady=10)

    b1.pack(side=TOP)
    draw_policy(c, xmdp1, 100, scale * 0.7, mdp1.get_states())
    draw_policy(c, xmdp2, 100, scale * 0.7, mdp2.get_states())
    draw_graph(c, xmdp1, 100, scale * 0.7, mdp1.get_states())
    draw_graph(c, xmdp2, 100, scale * 0.7, mdp2.get_states())
    draw_graph(c, xmdp3, 100, scale_mdp_3, mdp3.get_states())

    c.pack()
    top.mainloop()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
