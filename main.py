from tkinter import *

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# initialStateDistribution = [0.1, 0.05, 0.05, 0.2, 0.4, 0.2]
# initialStateId = np.random.choice(init_set_of_states).id
from mdp import Mdp
scale=125
xmdp1=50
xmdp2=750
mdp1 = Mdp(5, 1, [])
policy = mdp1.solve_mdp()
mdp2 = Mdp(5, 1,policy)




def draw_square(c, x, y, w, h, color):
    c.create_line(x, y, x + w, y, fill=color, width=3)
    c.create_line(x + w, y, x + w, y + h, fill=color, width=3)
    c.create_line(x + w, y + h, x, y + h, fill=color, width=3)
    c.create_line(x, y + h, x, y, fill=color, width=3)


def draw_rectangle(c, x, y, w, h, action,reward, color):
    if reward>0:
        if action == 'up':
            c.create_line(x, y, x + 0.5 * w, y, fill=color, width=3)
            c.create_line(x + 0.5 * w, y, x, y - 0.5*h, fill=color, width=3)
            c.create_line(x, y - 0.5*h, x - 0.5 * w, y, fill=color, width=3)
            c.create_line(x - 0.5 * w, y, x, y, fill=color, width=3)
            return
        if action == 'down':
            c.create_line(x, y, x + 0.5 * w, y, fill=color, width=3)
            c.create_line(x + 0.5 * w, y, x, y + 0.5*h, fill=color, width=3)
            c.create_line(x, y + 0.5*h, x - 0.5 * w, y, fill=color, width=3)
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
    temp_end_state = {}
    for s in set_of_states:
        if s.get_start():
            temp_init_state = s
        elif s.get_end():
            temp_end_state = s
        else:
            draw_square(c, x + s.x * scale, y + s.y * scale, scale, scale, 'black')
            c.create_text(x + s.x * scale + scale / 2, y + s.y * scale + scale / 2, text=s.get_value()['r'],
                          anchor='nw',
                          font='TkMenuFont', fill='black')

    draw_square(c, x + temp_init_state.x * scale, y + temp_init_state.y * scale, scale, scale, 'blue')
    c.create_text(x + temp_init_state.x * scale, y + temp_init_state.y * scale, text='Start', anchor='nw',
                  font='TkMenuFont', fill='blue')
    c.create_text(x + temp_init_state.x * scale + scale / 2, y + temp_init_state.y * scale + scale / 2,
                  text=temp_init_state.get_value()['r'], anchor='nw',
                  font='TkMenuFont', fill='black')
    draw_square(c, x + temp_end_state.x * scale, y + temp_end_state.y * scale, scale, scale, 'green')

    c.create_text(x + temp_end_state.x * scale, y + temp_end_state.y * scale, text='End ', anchor='nw',
                  font='TkMenuFont', fill='green')
    c.create_text(x + temp_end_state.x * scale + scale / 2, y + temp_end_state.y * scale + scale / 2,
                  text=temp_end_state.get_value()['r'], anchor='nw',
                  font='TkMenuFont', fill='black')


def draw_policy(c, x, y, scale, set_of_states):
    for s in set_of_states:
        draw_rectangle(c, x + s.x * scale + 0.5*scale, y + s.y * scale + 0.5*scale, 0.5*scale, 0.5*scale, s.value['a'],s.value['r'], 'pink')


def run_iteration(c):
    print('iteration wird ausgeführt')

    mdp1.run_iteration()
    mdp2.run_iteration()
    c.delete('all')
    draw_policy(c, xmdp1, 100, scale, mdp1.get_states())
    draw_policy(c, xmdp2, 100, scale, mdp2.get_states())
    draw_graph(c, xmdp1, 100, scale, mdp1.get_states())
    draw_graph(c, xmdp2, 100, scale, mdp2.get_states())

def run_transformation(c):
    mdp2.run_transformation()
    c.delete('all')
    draw_policy(c, xmdp1, 100, scale, mdp1.get_states())
    draw_policy(c, xmdp2, 100, scale, mdp2.get_states())
    draw_graph(c, xmdp1, 100, scale, mdp1.get_states())
    draw_graph(c, xmdp2, 100, scale, mdp2.get_states())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    top = Tk()
    top.geometry("1260x960")
    # creating a simple canvas
    c = Canvas(top, bg="white", height="960", width="1460")
    b1 = Button(top, text="Increment Iteration",
                command=lambda: run_transformation(c),
                activeforeground="red", activebackground="pink", pady=10)

    b1.pack(side=TOP)
    draw_policy(c, xmdp1, 100, scale, mdp1.get_states())
    draw_policy(c, xmdp2, 100, scale, mdp2.get_states())
    draw_graph(c, xmdp1, 100, scale, mdp1.get_states())
    draw_graph(c, xmdp2, 100, scale, mdp2.get_states())
    c.pack()
    top.mainloop()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
