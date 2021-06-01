class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.start = False
        self.end = False
        self.penalty = False
        self.value = {'a':'up','r':0}

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_start(self):
        return self.start

    def set_start(self, start):
        self.start = start

    def get_end(self):
        return self.end

    def set_end(self, end):
        self.end = end

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = {'a': value['a'],'r':round(value['r'], 6)}

    def set_penalty(self, penalty):
        self.penalty = penalty

    def get_penalty(self):
        return self.penalty


