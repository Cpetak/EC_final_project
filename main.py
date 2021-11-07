import numpy as np
import argparse
from scipy.stats import beta
import copy

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
    if sum(input_conc) == 0:
        return input_conc
    else:
        sum_input = input_conc-0.5
        x = sum_input * -args.alpha
        output = 1/(1 + np.exp(x))
        return output

def code_to_indx(args,mycode):
    x=int(mycode / args.grn_size)
    y=mycode - (x*args.grn_size)
    return(x,y)

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


    def eval_fitness(self, environment, args):
        if sum(self.phenotype) == 0:
            self.fitness = 0
        else:
            grn_out = np.asarray(self.phenotype)
            diff = np.abs(grn_out - environment).sum() # maximum is num_genes_consider
            self.fitness = (1-diff/args.num_genes_consider)**3 #TODO random network's fitness can be as high as 0.6 already!

def evolutionary_algorithm(args):

    fitness_over_time = []

    population = create_pop(args)
    envs = generate_optimum(args) # create optimal environments
    state = 0 # which environment are we living in

    for generation_num in range(args.num_generations):

        # mutation
        for p in population:
            sites = np.random.choice(args.grn_size*args.grn_size, args.mut_rate, replace=False)
            for s in sites:
                x,y=code_to_indx(args,s)
                p.grn[x][y]=np.random.normal(0,1) # mean = 0, standard deviation = 1

        # evaluation
        for i in range(len(population)):
            population[i].calculate_phenotype(args)
            population[i].eval_fitness(envs[state],args)

        # selection
        population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        #print([p.fitness for p in population])

        parents = population[:args.num_parents]
        if args.num_parents > args.pop_size/2:
            print("error, you are selecting too many parents")
            break
        else:
            kids = parents + parents # duplicate population, ie each parent has 1 kid
            additional_kids=np.random.choice(parents, args.pop_size-len(kids), replace=True) # then for the remainding kids (if there aren't any) randomly pick x parents and duplicate those again
            kids = kids + list(additional_kids)

        population = copy.deepcopy(kids)
        #print([p.fitness for p in population])

        # record keeping
        mean_f=np.mean([i.fitness for i in population])
        fitness_over_time.append(mean_f) # max fitness over time

    print(fitness_over_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-grn_size', type=int, default=20, help="GRN size")
    parser.add_argument('-max_iter', type=int, default=10, help="Maximum number of GRN updates")
    parser.add_argument('-pop_size', type=int, default=10, help="Population size")
    parser.add_argument('-alpha', type=float, default=10, help="Alpha for sigmoid function")
    parser.add_argument('-num_genes_consider', type=int, default=5, help="number of genes to consider in phenotype")
    parser.add_argument('-mut_rate', type=int, default=3, help="number of sites to mutate per individual")
    parser.add_argument('-num_generations', type=int, default=10, help="number of generations to run the experiment for")
    parser.add_argument('-num_parents', type=int, default=5, help="number of parents during truncation selection")

    args = parser.parse_args()

    print("running code")

    evolutionary_algorithm(args)
