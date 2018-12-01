class Result:
    def __init__(self,menorcaminho,valordocaminho,alpha,beta,ant_count,iterations,pheromone_evaporation_coefficient,pheromone_constant):
        self.menorcaminho = menorcaminho
        self.valordocaminho = valordocaminho
        self.alpha = alpha
        self.beta = beta
        self.ant_count = ant_count
        self.iterations = iterations
        self.pheromone_evaporation_coefficient = pheromone_evaporation_coefficient
        self.pheromone_constant = pheromone_constant
        self.filename = ''
        self.testname = ''
        self.standarddeviation = 0.0
