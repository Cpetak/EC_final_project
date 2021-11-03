import numpy as np
import argparse
from scipy.stats import beta

# Helper functions
def create_pop(args):
    pop = []

    for i in range(args.pop_size):
        new_agent = Individual(args)
        pop.append(new_agent)

    print("created pop")
    return(pop)

def my_sigmoid(args,input_conc):
    #adapted from wanger 2014
    sum_input = input_conc-0.5
    x = sum_input * -args.alpha
    output = 1/(1 + np.exp(x))
    return output

def generate_optimum(args):
    envs = []
    x = np.linspace(0,1,100)
    a, b = 2, 7
    y = beta.pdf(x, a, b) #using probability density function from scipy.stats
    y /= y.max()
    e=[]
    for i in range(args.num_genes_consider):
        t = y[int((100/args.num_genes_consider)*i)]
        t = np.around(t,2)
        e.append(t)
    envs.append(np.asarray(e))
    envs.append(np.asarray(e[::-1]))

    print("created envs")
    return(envs)

class Individual:
    def __init__(self,args):

        mu, sigma = 0, 1 # mean and standard deviation
        self.grn=np.random.normal(mu, sigma, (args.grn_size,args.grn_size))

        self.fitness = 0
        self.phenotype = np.zeros(args.num_genes_consider)
        self.complexity = 0

    def calculate_phenotype(self,args):
        step = lambda grn,input_conc: my_sigmoid(args,input_conc.dot(grn)) #a step is matrix multiplication followed by checking on sigmodial function
        input_conc = np.zeros(args.grn_size)
        input_conc[0] = 1 #starts with maternal factor switching on 1 gene
        e=0 #counter for state stability
        i=0 #counter for number of GRN updates
        input_concentrations=[] #stores input states for comparision to updated state

        while e < 1 and i < args.max_iter:
            input_concentrations.append(input_conc) #only maternal ON at first
            input_conc=step(self.grn,input_conc) # update protein concentrations
            input_conc=np.around(input_conc,2)
            if np.array_equal(input_concentrations[i], input_conc):
                e+=1
            i+=1

        if e != 0:
            self.phenotype = input_conc[-args.num_genes_consider:]
            self.complexity = i


    def eval_fitness(self):
        if sum(self.phenotype) == 0: #TODO: should I have a nonzero minimum sum concentration?
            self.fitness = 0
        else:
            self.fitness = 1


def evolutionary_algorithm(args):

    fitness_over_time=[]

    population=create_pop(args)
    envs = generate_optimum(args)
    for env in envs:
        print(env)

    for i in range(len(population)):
        population[i].calculate_phenotype(args)
        population[i].eval_fitness()
        print(population[i].phenotype)
        print(population[i].fitness)
        print(population[i].complexity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-grn_size', type=int, default=20, help="GRN size")
    parser.add_argument('-max_iter', type=int, default=10, help="Maximum number of GRN updates")
    parser.add_argument('-pop_size', type=int, default=10, help="Population size")
    parser.add_argument('-alpha', type=float, default=10, help="Alpha for sigmoid function")
    parser.add_argument('-num_genes_consider', type=int, default=5, help="number of genes to consider in phenotype")

    args = parser.parse_args()

    print("running code")

    evolutionary_algorithm(args)
