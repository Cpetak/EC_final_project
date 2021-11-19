# -*- coding: utf-8 -*-
"""EC_basic.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xkchY_NlzvGeAUS7cr601KCOJfLTZDV9

In the basic algorithm, a random number of positions will be selected based on the mutation rate and new weights will be randomly drawn from a Gaussian distribution with μ = 0, σ = 1 for those positions. Selection will be implemented as a simple truncation selection. Only children will be considered in the selection process (max_age = 1).

In this notebook also: 1) Possibility of selecting parents for the next generation without aging (max_age > generations), 2) Same but with aging (generation > max_age > 1)
"""

import torch
from tqdm import trange, tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

device="cuda"

#Helper functions

def to_im(ten):
    return ten.cpu().detach().clone().squeeze(1).numpy()


def dround(ten, digits):
    a = 10 ^ digits
    return torch.round(ten * a) / a


def fitness_function(pop, targ):
    return (1 - torch.abs(pop.squeeze(1) - targ)).sum(axis=1) # the smaller the difference, the higher the fitness


# Evolve
def evolutionary_algorithm(args):

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
        wandb.log({'average_complexity': args.max_iter-complexities.mean().item()}, commit=False)

        # Evaluate fitnesses
        phenos = state[:,:,:num_genes_fit]
        fitnesses = fitness_function(phenos, targs[curr_targ])
        cheaters = torch.where(complexities == 0)
        fitnesses[cheaters] = 0 # 0 fitness for non-converging ?? complexity part of fitness function, or fitness function computed thorughout the different states ??
        max_fits.append(fitnesses.max().item()) # keeping track of max fitness
        ave_fits.append(fitnesses.mean().item()) # keeping track of average fitness
        wandb.log({'max_fits': fitnesses.max().item()}, commit=False)
        wandb.log({'ave_fits': fitnesses.mean().item()}, commit=False)

        # Selection, the only part changed for tournament experiments
        winners = []
        for t in range(int(args.num_tournaments)):
          shuffled_idx = torch.randperm(args.pop_size, device="cuda")[:args.tournament_size] # select x number of individuals at random
          perm = torch.argsort(fitnesses[shuffled_idx], descending=True) # sort them based on fitnesses
          w_locs = shuffled_idx[perm[:args.num_tournament_winners]] # get parent locations
          winners.append(w_locs)

        winner_locs = torch.cat(winners, dim=0) #concat winners into 1 list
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
        wandb.log({'champions': state[winner_locs[0]].detach().clone().cpu().squeeze(0).numpy()}, commit=False)
        wandb.log({'best_grns': pop[winner_locs[0]].detach().clone().cpu()}, commit=False)

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
        wandb.log({'max_ages': ages.max().item()}, commit=False)
        wandb.log({'ave_ages': ages.max().item()}, commit=True)


        if gen % args.season_len == args.season_len - 1: # flip target
            curr_targ = (curr_targ + 1) % 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-grn_size', type=int, default=20, help="GRN size") # number of genes in the GRN
    parser.add_argument('-max_iter', type=int, default=10, help="Maximum number of GRN updates") # number of times gene concentrations are updated to get phenotype
    parser.add_argument('-pop_size', type=int, default=10, help="Population size")
    parser.add_argument('-alpha', type=int, default=10, help="Alpha for sigmoid function")
    parser.add_argument('-num_genes_consider', type=float, default=0.7, help="proportion of genes considered for fitness")
    parser.add_argument('-mut_rate', type=float, default=0.1, help="rate of mutation (i.e. number of genes to mutate)")
    parser.add_argument('-mut_size', type=float, default=0.5, help="size of mutation")
    parser.add_argument('-num_generations', type=int, default=10, help="number of generations to run the experiment for") # number of generations
    parser.add_argument('-max_age', type=int, default=5, help="max age at which individual is replaced by its kid")
    parser.add_argument('-season_len', type=int, default=5, help="number of generations between environmental flips")
    parser.add_argument('-proj', type=str, default="EC_final_project", help="Name of the project (for wandb)")
    parser.add_argument('-exp_type', type=str, default="BASIC", help="Name your experiment for grouping")
    #parser.add_argument('-rep', type=str, default="1", help="ID of replicate")

    parser.add_argument('-tournament_size', type=int, default=5, help="Number of individuals competeing in a tournament")
    parser.add_argument('-num_tournament_winners', type=int, default=2, help="Number of individuals winning in a tournament")

    args = parser.parse_args()

    print("running code")

    args.num_tournaments = args.pop_size / args.num_tournament_winners

    if args.pop_size % args.num_tournament_winners != 0:
      print("Error: select different num_tournament_winners")
      break

    #tag = conf2tag(vars(args))
    wandb.init(config=args, project=args.proj, group=args.exp_type)

    evolutionary_algorithm(args)





#plt.figure(figsize=(18, 10))
#plt.plot(max_fits)
#plt.tight_layout()
#plt.title("max fitness is "+str(N*M))
#plt.show()

#plt.plot(ave_complex)
#plt.tight_layout()
#plt.title("max complexity is "+str(lifetime))
#plt.show()

#plt.figure(figsize=(18, 10))
#plt.plot(max_ages)
#plt.tight_layout()
#plt.show()

#c = np.stack(champions)  # + [targ.cpu().detach().clone().squeeze(0).numpy()])
#d = c.mean(axis=0).argsort()
#plt.figure(figsize=(18, 10))
# plt.imshow(c.T, interpolation="nearest")
#plt.imshow(c[:, d].T, interpolation="nearest")
#plt.ylabel("GENE")
#plt.xlabel("Epoch")
#plt.tight_layout()
#plt.colorbar()
#plt.show()
