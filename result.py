class Result:
    def __init__(self, f1, acc_mean, acc_std, j_mean, j_std):
        self.f1 = f1
        self.acc_mean = acc_mean
        self.acc_std = acc_std
        self.j_mean = j_mean
        self.j_std = j_std
        self.num_hidden = 0
        self.nodes_per_hidden = 0
        self.learning_rate = 0
        self.fator_reg = 0
        self.filename = ''