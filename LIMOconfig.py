class LIMOconfiguration:
    def __init__(self):
        self.PREFERENCE_SET_SIZE = 20  # size of the set of preferences. This is also the number of dots in the plot
        self.LATEX_PRINT_MODE = 'single'  # options single|multiple
        self.NOISE_FACTOR = 0.2  # the noise-factor. It defines the uncertainty under which the agent acts. 0.2 means
        # that with 20% chance the agents doesn't do what he was supposed to do
