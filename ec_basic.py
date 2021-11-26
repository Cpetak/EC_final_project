import torch
from tqdm import trange, tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

import pickle
from copy import deepcopy
import wandb
from datetime import date
from pathlib import Path

device="cuda"

#Helper functions

def to_im(ten):
    return ten.cpu().detach().clone().squeeze(1).numpy()


def dround(ten, digits):
    a = 10 ^ digits
    return torch.round(ten * a) / a


def fitness_function(pop, targ):
    return (1 - torch.abs(pop.squeeze(1) - targ)).sum(axis=1) # the smaller the difference, the higher the fitness

class FakeArgs:
    """
    A simple class imitating the args namespace
    """

    def __repr__(self):
        attrs = vars(self)
        return "\n".join([f"{k}: {v}" for k, v in attrs.items()])

def prepare_run(entity, project, args, folder_name="results"):
    import wandb

    #folder = get_folder()
    #args.location = get_location()

    run = wandb.init(config=args, entity=entity, project=project)

    today = date.today().strftime("%d-%m-%Y")
    folder = Path(folder_name) / run.name
    folder.mkdir(parents=True, exist_ok=True)

    return run, folder

# Evolve
def evolutionary_algorithm(args, title, folder):

    #Setting up

    pop = torch.randn((args.pop_size, args.grn_size, args.grn_size)).to(device) # create population of random GRNs
    targ = torch.randint(2, size=(1, args.grn_size)).to(device) # create a random target, binary (for better visualisation)
    num_genes_fit=int(args.num_genes_consider*args.grn_size)
    targ = targ[:,:num_genes_fit] # create targets only for relevant genes
    targs = [targ, 1 - targ.detach().clone()] # alternative target is the exact opposite
    curr_targ = 0 # ID of which target is the current one

    # Keeping track

    ages = torch.zeros(args.pop_size)
    max_fits = []
    ave_fits = []
    ave_complex = []
    champions = []
    max_ages = []
    ave_ages = []
    best_grns = []
    mut_rates_t = []
    mut_sizes_t = []

    diversities = []

    # create all possible masks for 2-point crossover
    if args.crossover == "twopoint":
        idxs = np.array(list(combinations(range(0, args.grn_size+1),2)))
        masks = torch.zeros(len(idxs),args.grn_size,args.grn_size, device="cuda")
        for i,(start,end) in enumerate(idxs):
          masks[i,:,start:end] = 1
        antimasks = 1 - masks

    for gen in trange(args.num_generations):

        complexities = torch.zeros(args.pop_size)

        # Generating phenotypes
        state = torch.zeros(args.pop_size, 1, args.grn_size).to(device)
        state[:, :, 0] = 1.0 # create input to the GRNs

        state_before = torch.zeros(args.pop_size, 1, args.grn_size).to(device) # keeping track of the last state
        for l in range(args.max_iter):
          state = torch.matmul(state, pop) # each matrix in the population is multiplied
          state = state * args.alpha
          state = torch.sigmoid(state) # after which it is put in a sigmoid function to get the output, by default alpha = 1 which is pretty flat, so let's use alpha > 1 (wagner uses infinite) hence the above multiplication
          # state = dround(state, 2)
          diffs=torch.abs(state_before - state).sum(axis=(1,2))
          which_repeat = torch.where(diffs == 0)
          complexities[which_repeat] += 1
          state_before = state

        ave_complex.append(args.max_iter-complexities.mean().item()) # 0 = never converged, the higher the number the earlier it converged so true "complexity" is inverse of this value
        run.log({'average_complexity': args.max_iter-complexities.mean().item()}, commit=False)

        # Evaluate fitnesses
        phenos = state[:,:,:num_genes_fit]
        fitnesses = fitness_function(phenos, targs[curr_targ])
        cheaters = torch.where(complexities == 0)
        fitnesses[cheaters] = 0 # 0 fitness for non-converging ?? complexity part of fitness function, or fitness function computed thorughout the different states ??
        max_fits.append(fitnesses.max().item()) # keeping track of max fitness
        ave_fits.append(fitnesses.mean().item()) # keeping track of average fitness
        run.log({'max_fits': fitnesses.max().item()}, commit=False)
        run.log({'ave_fits': fitnesses.mean().item()}, commit=False)

        # Adaptive mut rate
        if args.adaptive_mut:
            # where ages == 0 -> they where born in the previous generation and haven't been modified yet but fitnesses are calculated
            pfits = fitnesses[torch.where(ages > 0)] #fitnesses of parents
            cfits = fitnesses[torch.where(ages == 0)] #fitnesses of children

            if len(pfits) > 0 and len(cfits) > 0:
              p_min_fitness=torch.min(pfits) # min fitness of parents
              num_fit_children = len(cfits[torch.where(cfits > p_min_fitness)]) # number of children fitter than worse parent

              if num_fit_children > (len(cfits)/5): # if more than 1/5 of children are better than worse parent -> increase learning rate as we are still far from optimum
                #print("increase learning rate")
                args.mut_rate += args.meta_mut_rate
                args.mut_size += args.meta_mut_rate

              if num_fit_children < (len(cfits)/5): # if less than 1/5 of children are better than worse parent -> decrease learning rate as we are close to optimum
                #print("decrease learning rate")
                if args.mut_rate - args.meta_mut_rate > 0:
                    args.mut_rate -= args.meta_mut_rate
                if args.mut_size - args.meta_mut_rate > 0:
                    args.mut_size -= args.meta_mut_rate

            mut_rates_t.append(args.mut_rate)
            run.log({'mut_rates_t': args.mut_rate}, commit=False)
            mut_sizes_t.append(args.mut_size)
            run.log({'mut_sizes_t': args.mut_size}, commit=False)

        # Selection
        perm = torch.argsort(fitnesses, descending=True)
        parent_locs = perm[:args.truncation_size] # location of top x parents in the array of individuals
        children_locs = perm[args.truncation_size:] # location of individuals that won't survive and hence will be replaced by others' children

        champions.append(state[perm[0]].detach().clone().cpu().squeeze(0).numpy()) # keeping tract of best solution's output
        best_grns.append(pop[perm[0]].detach().clone().cpu()) # keeping tract of best solution
        #run.log({'champions': state[perm[0]].detach().clone().cpu().squeeze(0).numpy()}, commit=False)
        #run.log({'best_grns': pop[perm[0]].detach().clone().cpu()}, commit=False)

        ages[parent_locs] += 1 # updating the ages of the individuals
        ages[children_locs] = 0

        parents = pop[parent_locs].detach().clone() # access parents' matricies
        num_child = int(args.pop_size/args.truncation_size) - 1
        children = parents.repeat([num_child, 1, 1]) # create copies of parents

        # Mutation
        num_genes_mutate = int(args.grn_size*args.grn_size*len(children) * args.mut_rate)
        mylist = torch.zeros(args.grn_size*args.grn_size*len(children), device="cuda")
        mylist[:num_genes_mutate] = 1
        shuffled_idx = torch.randperm(args.grn_size*args.grn_size*len(children), device="cuda")
        mask = mylist[shuffled_idx].reshape(len(children),args.grn_size,args.grn_size) #select genes to mutate
        children = children + (children*mask)*torch.randn(size=children.shape, device="cuda") * args.mut_size  # mutate only children only at certain genes

        pop[children_locs] = children # put children into population

        # Crossover, between kids (concenptually the same as if I first did the crossover, then the mutation), otherwise same as basic model! parents can crossover too...
        if args.crossover != "NO":
            cpairs=torch.randperm(args.pop_size, device="cuda")[:args.num_crossover] #create kid pairs, num_crossover has to be divisibale by 2

            if args.crossover == "twopoint":
                random_mask_pos = torch.randperm(len(masks))[:int(args.num_crossover/2)] #get a random set of masks
                mymasks=masks[random_mask_pos]
                myantimasks=antimasks[random_mask_pos]

            if args.crossover == "uniform":
                all_col = torch.arange(args.grn_size, device="cuda")
                y=all_col.repeat(int(args.num_crossover/2),1)
                indices = torch.argsort(torch.rand(*y.shape), dim=-1)
                result = y[torch.arange(y.shape[0]).unsqueeze(-1), indices] #create random permutation of column orders

                mymasks = torch.where(result>args.grn_size/2, 0, 1) #make it into a 2D mask
                mymasks=mymasks.repeat(1,args.grn_size).reshape(int(args.num_crossover/2),args.grn_size,args.grn_size)
                myantimasks = torch.where(result<=args.grn_size/2, 0, 1) #make inverse into a 2D mask
                myantimasks=myantimasks.repeat(1,args.grn_size).reshape(int(args.num_crossover/2),args.grn_size,args.grn_size)

            n1=pop[cpairs[int(len(cpairs)/2):]] * mymasks + pop[cpairs[:int(len(cpairs)/2)]] * myantimasks # first cpair/2 individuals in cpairs, after crossover
            n2=pop[cpairs[:int(len(cpairs)/2)]] * mymasks + pop[cpairs[int(len(cpairs)/2):]] * myantimasks # second cpair/2 individuals in cpairs, after crossover
            all_pop=torch.arange(args.pop_size, device="cuda")
            not_crossed = [i for i in all_pop if i not in cpairs] # plus the individuals left out of crossover!!
            not_crossed_mats = pop[torch.stack(not_crossed,0)]
            new_pop = torch.cat((n1, n2, not_crossed_mats), 0)
            pop=new_pop

        # Dying due to old age
        old_locs = torch.where(ages >= args.max_age) # get location of old individuals

        if len(old_locs[0]) != 0:
          ages[old_locs] = 0 #reset age

          old_inds = pop[old_locs] # get old individuals' matrices

          num_genes_mutate = int(args.grn_size*args.grn_size*len(old_inds) * args.mut_rate)
          mylist = torch.zeros(args.grn_size*args.grn_size*len(old_inds), device="cuda")
          mylist[:num_genes_mutate] = 1
          shuffled_idx = torch.randperm(args.grn_size*args.grn_size*len(old_inds), device="cuda")
          mask = mylist[shuffled_idx].reshape(len(old_inds),args.grn_size,args.grn_size) #select genes to mutate
          old_inds = old_inds + (old_inds*mask)*torch.randn(size=old_inds.shape, device="cuda") * args.mut_size

          pop[old_locs] = old_inds # mutate old individual -> new child

        max_ages.append(ages.max().item())
        ave_ages.append(ages.mean().item())
        run.log({'max_ages': ages.max().item()}, commit=False)
        run.log({'ave_ages': ages.mean().item()}, commit=False)

        d=torch.mean(torch.std(pop,unbiased=False, dim=0))
        diversities.append(d)
        run.log({'diversities': d}, commit=True)

        if gen % args.season_len == args.season_len - 1: # flip target
            curr_targ = (curr_targ + 1) % 2

    stats = {}
    stats["max_fits"] = max_fits
    stats["ave_fits"] = ave_fits
    stats["ave_complex"] = ave_complex
    stats["champions"] = champions
    stats["max_ages"] = max_ages
    stats["ave_ages"] = ave_ages
    stats["best_grns"] = best_grns
    stats["diversities"] = diversities
    with open(f"{folder}/basic_{title}.pkl", "wb") as f:
        pickle.dump(stats, f)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()

    args = FakeArgs()

    #parser.add_argument('-grn_size', type=int, default=50, help="GRN size") # number of genes in the GRN
    args.grn_size = 50
    #parser.add_argument('-pop_size', type=int, default=1000, help="Population size")
    args.pop_size = 1000
    #parser.add_argument('-alpha', type=int, default=10, help="Alpha for sigmoid function")
    args.alpha = 10
    #parser.add_argument('-num_genes_consider', type=float, default=0.5, help="proportion of genes considered for fitness")
    args.num_genes_consider = 0.5
    #parser.add_argument('-mut_rate', type=float, default=0.1, help="rate of mutation (i.e. number of genes to mutate)")
    args.mut_rate = 0.1
    #parser.add_argument('-mut_size', type=float, default=0.5, help="size of mutation")
    args.mut_size = 0.5
    #parser.add_argument('-num_generations', type=int, default=100000, help="number of generations to run the experiment for") # number of generations
    args.num_generations = 50000
    #parser.add_argument('-truncation_prop', type=float, default=0.2, help="proportion of individuals selected for reproduction")
    args.truncation_prop = 0.2
    #parser.add_argument('-max_age', type=int, default=30, help="max age at which individual is replaced by its kid")
    args.max_age = 30
    #parser.add_argument('-season_len', type=int, default=100, help="number of generations between environmental flips")
    args.season_len = 100
    #parser.add_argument('-proj', type=str, default="EC_final_project", help="Name of the project (for wandb)")
    args.proj = "EC_final_project"
    #parser.add_argument('-exp_type', type=str, default="BASIC", help="Name your experiment for grouping")
    args.exp_type = "BASIC"

    #parser.add_argument('-crossover', type=str, default="NO", help="Options: NO, uniform, twopoint")
    args.crossover = "NO"
    #parser.add_argument('-crossover_freq', type=float, default=0.5, help="number of individuals that will undergo crossover")
    args.crossover_freq = 0.5
    #parser.add_argument('-adaptive_mut', type=bool, default=False, help="if you want adaptive mutation rate")
    args.adaptive_mut = False
    #parser.add_argument('-meta_mut_rate', type=float, default=0.01, help="how much you increase or decrease mut_size and mut_rate")
    args.meta_mut_rate = 0.01

    #args = parser.parse_args()

    print("running code")

    args.max_iter = int(3*args.grn_size) # "Maximum number of GRN updates") # number of times gene concentrations are updated to get phenotype

    args.truncation_size=int(args.truncation_prop*args.pop_size)

    #TO CHANGE
    args.adaptive_mut = True
    args.meta_mut_rate = 0.001
    print(args)

    args.num_crossover = int(args.crossover_freq * args.pop_size) #how many individuals will be involved in crossover

    assert (
        args.num_crossover % 2 == 0
    ), f"Error: select different crossover_freq: received {args.num_crossover}"
    assert (
        args.pop_size % args.truncation_size == 0
    ), "Error: select different trunction_prop, received {args.pops_size}"

    
    run, folder = prepare_run("molanu", args.proj, args)

    evolutionary_algorithm(args, f"{args.num_genes_consider}", folder)

    #for num_genes_consider in [0.2, 0.5, 0.8]:
    #for season_len in [50, 100, 500]:
        #args = deepcopy(defaults)
        #args.season_len = season_len
        #print(args)
        #evolutionary_algorithm(args, f"{season_len}_{rep}_", folder)
    #for max_age in [0, 30, 100000000000]:
        #args = deepcopy(defaults)
        #args.max_age = max_age
        #print(args)
        #evolutionary_algorithm(args, f"{max_age}_{rep}_", folder)

