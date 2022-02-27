class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.start = False
        self.finish = False
        self.value = {'a': 'up', 'r': 0}
        self.add_values = []
        self.solid = False
        self.frequency = 0

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_start(self):
        return self.start

    def set_start(self, start):
        self.start = start

    def get_finish(self):
        return self.finish

    def set_finish(self, finish):
        self.finish = finish

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = {'a': value['a'], 'r': value['r']}

    def clear_value(self):
        self.value = {'a': 'up', 'r': 0}
        self.add_values = []

    def set_add_values(self, add_values):
        # print('same value for a state, ',add_values)
        self.add_values = add_values

    def get_add_values(self):
        return self.add_values

    def get_solid(self):
        return self.solid

    def set_solid(self, solid):
        self.solid = solid

    def increase_frequency(self, freq):
        self.set_frequency(self.get_frequency() + freq)

    def set_frequency(self, freq):
        self.frequency = freq

    def get_frequency(self):
        return self.frequency

    def has_coord(self, coord):
        return coord['x'] == self.get_x() and coord['y'] == self.get_y()

    def is_terminal(self):
        return self.get_x() == 'end' and self.get_y() == 'end'
