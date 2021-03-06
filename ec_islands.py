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
    #island 0: static 1, island 1: static 2, island 2: variable environment, flipping between static 1 and static 2
    pop = torch.randn((args.pop_size, args.grn_size, args.grn_size)).to(device) # create population of random GRNs
    targ = torch.randint(2, size=(1, args.grn_size)).to(device) # create a random target, binary (for better visualisation)
    num_genes_fit=int(args.num_genes_consider*args.grn_size)
    targ = targ[:,:num_genes_fit] # create targets only for relevant genes
    targs = [targ, 1 - targ.detach().clone()] # alternative target is the exact opposite
    curr_targ = 0 # ID of which target is the current one

    static1=targs[0] # static target for island 0
    static2=targs[1] # static target for island 1

    islands = torch.zeros(args.pop_size, device="cuda")
    islands[int(args.pop_size/3):int(2*(args.pop_size/3))] = 1
    islands[int(2*(args.pop_size/3)):] = 2

    # Keeping track

    ages = torch.zeros(args.pop_size)

    c_island0 = []
    c_island1 = []
    c_island2 = []

    mf_island0 = []
    mf_island1 = []
    mf_island2 = []

    af_island0 = []
    af_island1 = []
    af_island2 = []


    champions = []
    max_ages = []
    ave_ages = []
    best_grns = []

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

        c_island0.append(args.max_iter-complexities[torch.where(islands==0)].mean().item()) # 0 = never converged, the higher the number the earlier it converged so true "complexity" is inverse of this value
        c_island1.append(args.max_iter-complexities[torch.where(islands==1)].mean().item())
        c_island2.append(args.max_iter-complexities[torch.where(islands==2)].mean().item())

        run.log({'c_island0': args.max_iter-complexities[torch.where(islands==0)].mean().item()}, commit=False)
        run.log({'c_island1': args.max_iter-complexities[torch.where(islands==1)].mean().item()}, commit=False)
        run.log({'c_island2': args.max_iter-complexities[torch.where(islands==2)].mean().item()}, commit=False)

        # Evaluate fitnesses, based on island they belong to
        phenos = state[:,:,:num_genes_fit]
        fitnesses = fitness_function(phenos, targs[curr_targ])

        fitnesses[torch.where(islands == 0)] = fitness_function(phenos[torch.where(islands == 0)], static1) # fitnesses for island 0
        fitnesses[torch.where(islands == 1)] = fitness_function(phenos[torch.where(islands == 1)], static2) # fitnesses for island 1

        cheaters = torch.where(complexities == 0)
        fitnesses[cheaters] = 0 # 0 fitness for non-converging ?? complexity part of fitness function, or fitness function computed thorughout the different states ??

        mf_island0.append(fitnesses[torch.where(islands==0)].max().item()) # keeping track of max fitness
        mf_island1.append(fitnesses[torch.where(islands==1)].max().item())
        mf_island2.append(fitnesses[torch.where(islands==2)].max().item())

        af_island0.append(fitnesses[torch.where(islands==0)].mean().item()) # keeping track of average fitness
        af_island1.append(fitnesses[torch.where(islands==1)].mean().item())
        af_island2.append(fitnesses[torch.where(islands==2)].mean().item())

        run.log({'mf_island0': fitnesses[torch.where(islands==0)].max().item()}, commit=False)
        run.log({'mf_island1': fitnesses[torch.where(islands==1)].max().item()}, commit=False)
        run.log({'mf_island2': fitnesses[torch.where(islands==2)].max().item()}, commit=False)
        run.log({'af_island0': fitnesses[torch.where(islands==0)].mean().item()}, commit=False)
        run.log({'af_island1': fitnesses[torch.where(islands==1)].mean().item()}, commit=False)
        run.log({'af_island2': fitnesses[torch.where(islands==2)].mean().item()}, commit=False)


        # Selection, independent for each island
        all_pop=torch.arange(args.pop_size, device="cuda")
        parent_locs=[]
        children_locs=[]

        for i in range(3): #for each island
          ori_indx = all_pop[torch.where(islands == i)]
          fits=fitnesses[ori_indx]
          perm=torch.argsort(fits, descending=True)
          tparent_locs=ori_indx[perm][:args.truncation_size] #where in original pop list are the selected parents for this island
          tchildren_locs=ori_indx[perm][args.truncation_size:]

          champions.append(state[tparent_locs[0]].detach().clone().cpu().squeeze(0).numpy()) #island 0, then 1, then 2
          best_grns.append(pop[tparent_locs[0]].detach().clone().cpu())
          #run.log({'champions': state[tparent_locs[0]].detach().clone().cpu().squeeze(0).numpy()}, commit=False)
          #run.log({'best_grns': pop[tparent_locs[0]].detach().clone().cpu()}, commit=False)

          parent_locs.append(tparent_locs)
          children_locs.append(tchildren_locs)

        parent_locs = torch.cat(parent_locs, dim=0) # all parent locs selected independently in each island
        children_locs = torch.cat(children_locs, dim=0)

        ages[parent_locs] += 1 # updating the ages of the individuals
        ages[children_locs] = 0

        parents = pop[parent_locs].detach().clone() # access parents' matricies
        #truncation_size=num individuals selected in each island
        num_child = int(args.pop_size/(args.truncation_size*3)) - 1 # num children to be created for all islands
        parent_islands = islands[parent_locs] # keep track of where parents come from
        children = parents.repeat([num_child, 1, 1]) # create copies of parents

        # Mutation
        num_genes_mutate = int(args.grn_size*args.grn_size*len(children) * args.mut_rate)
        mylist = torch.zeros(args.grn_size*args.grn_size*len(children), device="cuda")
        mylist[:num_genes_mutate] = 1
        shuffled_idx = torch.randperm(args.grn_size*args.grn_size*len(children), device="cuda")
        mask = mylist[shuffled_idx].reshape(len(children),args.grn_size,args.grn_size) #select genes to mutate
        children = children + (children*mask)*torch.randn(size=children.shape, device="cuda") * args.mut_size  # mutate only children only at certain connections

        pop[children_locs] = children # put children into population
        islands[children_locs] = parent_islands.repeat(num_child) #update islands as parents.repeat creates kid for parent 1, then parent 2 etc then parent 1 again for n times

        #will be needed for both crossover and migration
        all_i=torch.arange(len(islands), device="cuda")
        island0=all_i[torch.where(islands==0)]
        island1=all_i[torch.where(islands==1)]
        island2=all_i[torch.where(islands==2)]

        # Crossover, between individuals of the same island
        if args.crossover == "twopoint":
            cpairs0=torch.randperm(int(args.pop_size/3), device="cuda")[:args.num_crossover] #num_crossover is the number of individuals involved in crossovers for each island, has to be divisible by 2
            cpairs1=torch.randperm(int(args.pop_size/3), device="cuda")[:args.num_crossover]
            cpairs2=torch.randperm(int(args.pop_size/3), device="cuda")[:args.num_crossover]
            cpairs = torch.cat((island0[cpairs0][:int(args.num_crossover/2)], island1[cpairs1][:int(args.num_crossover/2)], island2[cpairs2][:int(args.num_crossover/2)],island0[cpairs0][int(args.num_crossover/2):],island1[cpairs1][int(args.num_crossover/2):], island2[cpairs2][int(args.num_crossover/2):]), 0)

            random_mask_pos = torch.randperm(len(masks))[:int((args.num_crossover*3)/2)] #get a random set of masks
            mymasks=masks[random_mask_pos]
            myantimasks=antimasks[random_mask_pos]

            n1=pop[cpairs[int(len(cpairs)/2):]] * mymasks + pop[cpairs[:int(len(cpairs)/2)]] * myantimasks # first cpair/2 individuals in cpairs, after crossover
            n2=pop[cpairs[:int(len(cpairs)/2)]] * mymasks + pop[cpairs[int(len(cpairs)/2):]] * myantimasks # second cpair/2 individuals in cpairs, after crossover
            all_pop=torch.arange(args.pop_size, device="cuda")
            not_crossed = [i for i in all_pop if i not in cpairs] # plus the individuals left out of crossover!!
            not_crossed_mats = pop[torch.stack(not_crossed,0)]
            new_pop = torch.cat((n1, n2, not_crossed_mats), 0)
            pop=new_pop

        # Migration
        inds_to_move0 = torch.randperm(len(island0), device="cuda")[:args.num_migr*2] # select random individuals that will migrate
        inds_to_move1 = torch.randperm(len(island1), device="cuda")[:args.num_migr*2]
        inds_to_move2 = torch.randperm(len(island2), device="cuda")[:args.num_migr*2]

        #moving between 0 and 1
        new_islands=torch.clone(islands)
        new_islands[island0[inds_to_move0[:args.num_migr]]] = islands[island1[inds_to_move1[:args.num_migr]]]
        new_islands[island1[inds_to_move1[:args.num_migr]]] = islands[island0[inds_to_move0[:args.num_migr]]]
        islands = new_islands
        # moving between 0 and 2
        new_islands=torch.clone(islands)
        new_islands[island0[inds_to_move0[args.num_migr:]]] = islands[island2[inds_to_move2[:args.num_migr]]]
        new_islands[island2[inds_to_move2[:args.num_migr]]] = islands[island0[inds_to_move0[args.num_migr:]]]
        islands = new_islands
        # moving between 1 and 2
        new_islands=torch.clone(islands)
        new_islands[island1[inds_to_move1[args.num_migr:]]] = islands[island2[inds_to_move2[args.num_migr:]]]
        new_islands[island2[inds_to_move2[args.num_migr:]]] = islands[island1[inds_to_move1[args.num_migr:]]]
        islands = new_islands

        # Dying due to old age, nothing changes here with island model
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
    stats["c_island0"] = c_island0
    stats["c_island1"] = c_island1
    stats["c_island2"] = c_island2
    stats["mf_island0"] = mf_island0
    stats["mf_island1"] = mf_island1
    stats["mf_island2"] = mf_island2
    stats["af_island0"] = af_island0
    stats["af_island1"] = af_island1
    stats["af_island2"] = af_island2
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

    #parser.add_argument('-crossover', type=str, default="NO", help="Options: NO, twopoint")
    args.crossover = "NO"
    #parser.add_argument('-crossover_freq', type=float, default=0.5, help="prop of individuals involved in crossover in each island")
    args.crossover_freq = 0.3
    #parser.add_argument('-migration_rate', type=float, default=0.1, help="Options: NO, twopoint") # prop of individuals that move between any two islands, 0.3 means 1/3 of the population migrates into 1 island, 1/3 into the other island
    args.migration_rate = 0.1

    #args = parser.parse_args()

    print("running code")

    args.max_iter = int(3*args.grn_size) # "Maximum number of GRN updates") # number of times gene concentrations are updated to get phenotype

    #TO CHANGE
    args.crossover = "NO"
    args.crossover_freq = 0.3
    args.migration_rate = 0.1
    print(args)

    args.num_crossover = int(args.crossover_freq * (args.pop_size/3)) #num_crossover is the number of individuals involved in crossovers for each island, has to be divisible by 2
    args.truncation_size=int(args.truncation_prop*int(args.pop_size/3)) #num individuals selected in each island
    args.num_migr=int(args.pop_size/3 * args.migration_rate)

    assert (
        args.num_crossover % 2 == 0
    ), f"Error: select different crossover_freq: received {args.num_crossover}"
    assert (
        int(args.pop_size/3) % args.truncation_size == 0
    ), f"Error: select different trunction_prop, received {args.pop_size}"
    assert (
        args.pop_size % 3 == 0
    ), f"Error: select different pop_size, received {args.pop_size}"


    run, folder = prepare_run("molanu", args.proj, args)

    evolutionary_algorithm(args, f"{args.num_genes_consider}", folder)
