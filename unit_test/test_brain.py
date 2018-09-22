class TestBrain:
    def __init__(self, mode):
        self.mode = mode

    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y

    def predict(self, input_list):
        return_list = []
        for i in range(len(input_list[0])):
            choices = []
            for j in range(65):
                choices.append(0.1)

            if self.mode == "brain":
                choices[i] = input_list[0][i] + input_list[1][i]
                return_list.append(choices)
            else:
                choices[i+1] = 2
                return_list.append(choices)
        return return_list

    def fit(self, x, y, verbose=1, epochs=1):
        self.x = x
        self.y = y

        return 1