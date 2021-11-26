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
    fit_stds = []

    diversities = []

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

        # Selection, the only part changed for fitness proportional selection
        if args.ranked:
          sorted_f = torch.argsort(fitnesses, descending=True)
          ranks = torch.arange(args.pop_size, device="cuda")
          ranks[sorted_f] = args.pop_size - torch.arange(args.pop_size, device="cuda")
          f_props = ranks / sum(ranks)
        else:
          if sum(fitnesses) > 0:
            f_props = fitnesses / sum(fitnesses)
          else:
            f_props = fitnesses

        fit_stds.append(torch.std(f_props, unbiased=False))
        run.log({'fit_stds': torch.std(f_props, unbiased=False)}, commit =False)

        winner_locs=torch.multinomial(f_props, args.pop_size, replacement=True)
        uni_winner_locs, counts = torch.unique(winner_locs,sorted=True,return_counts=True) #select winners, counted only once - these will be included in the next generation unchanged
        counts = counts - 1
        rest_winner_locs = uni_winner_locs.repeat_interleave(counts) #this is how many offspring each winner has  - i.e. mutated copies

        #uni_winner_locs = where individuals survive to next generation
        #rest_winner_locs = how many kids each winner has

        all_pop=torch.arange(args.pop_size, device="cuda")
        children_l = [i for i in all_pop if i not in winner_locs]
        children_locs = torch.stack(children_l,0) # location of individuals that won't survive OR reproduce and hence will be replaced by others' children

        champions.append(state[winner_locs[0]].detach().clone().cpu().squeeze(0).numpy()) # keeping tract of best solution
        best_grns.append(pop[winner_locs[0]].detach().clone().cpu()) # keeping tract of best solution
        run.log({'champions': state[winner_locs[0]].detach().clone().cpu().squeeze(0).numpy()}, commit=False)
        run.log({'best_grns': pop[winner_locs[0]].detach().clone().cpu()}, commit=False)

        #uni_winner_locs = will age because those survive to the next generation
        #children_locs = is where the kids are placed
        ages[uni_winner_locs] += 1 # updating the ages of the individuals
        ages[children_locs] = 0

        parents = pop[rest_winner_locs].detach().clone() # access parents' matricies, individuals that not only survive but also reproduce

        # Mutation
        num_genes_mutate = int(args.grn_size*args.grn_size*len(children_locs) * args.mut_rate)
        mylist = torch.zeros(args.grn_size*args.grn_size*len(children_locs), device="cuda")
        mylist[:num_genes_mutate] = 1
        shuffled_idx = torch.randperm(args.grn_size*args.grn_size*len(children_locs), device="cuda")
        mask = mylist[shuffled_idx].reshape(len(children_locs),args.grn_size,args.grn_size) #select genes to mutate
        children = parents + (parents*mask) * torch.randn(size=parents.shape, device="cuda") * args.mut_size  # mutate only children only at certain genes

        pop[children_locs] = children # put children into population

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
    #parser.add_argument('-max_age', type=int, default=30, help="max age at which individual is replaced by its kid")
    args.max_age = 30
    #parser.add_argument('-season_len', type=int, default=100, help="number of generations between environmental flips")
    args.season_len = 100
    #parser.add_argument('-proj', type=str, default="EC_final_project", help="Name of the project (for wandb)")
    args.proj = "EC_final_project"

    #parser.add_argument('-ranked', type=bool, default=False, help="Wether or not to use rank-based proportional selection")
    args.ranked = False

    #args = parser.parse_args()

    print("running code")

    args.max_iter = int(3*args.grn_size) # "Maximum number of GRN updates") # number of times gene concentrations are updated to get phenotype

    #TO CHANGE
    args.ranked = False
    print(args)

    run, folder = prepare_run("molanu", args.proj, args)

    evolutionary_algorithm(args, f"{args.num_genes_consider}", folder)
