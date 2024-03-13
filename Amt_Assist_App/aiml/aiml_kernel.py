import aiml
import time
time.clock = time.time

class aiml_kernel:
    def __init__(self, aiml_files):
        self.kernel = aiml.Kernel()
        self.load_aiml(aiml_files)

    def load_aiml(self, aiml_files):
        for file in aiml_files:
            self.kernel.learn(file)

    def respond(self, user_input):
        return self.kernel.respond(user_input)
