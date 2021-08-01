class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.start = False
        self.end = False
        self.penalty = False
        self.value = {'a':'up','r':0}
        self.add_values=[]
        self.solid=False

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
        self.value = {'a': value['a'],'r':round(value['r'], 10)}
    def set_add_values(self,add_values):
       # print('same value for a state, ',add_values)
        self.add_values=add_values
    def get_add_values(self):
        return self.add_values

    def set_penalty(self, penalty):
        self.penalty = penalty

    def get_penalty(self):
        return self.penalty

    def get_solid(self):
        return self.solid

    def set_solid(self,solid):
        self.solid=solid

