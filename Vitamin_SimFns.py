'''
 Math for Parents

Problem: 
 A bottle of 180 children's vitamins is divided into three colors (orange, pink, and purple) 
 and four animal shapes (cat, lion, hippo, elephant). Every day, 2 children are presented a selection 
 of approximately 10 vitamins, poured from the top of the jar, from which each selects 2 vitamins, for 
 a total of 4 eaten per day. (The unselected vitamins are returned to the jar.)
 
 If Child A only wants purple vitamins (without regard to shape) and Child B only wants cat 
 vitamins (without regard to color), what are the odds each will be pleased with the selection on
 - Day 1?
 - Day 20? 
 - Day 45? 
 
 For extra credit, at what point in the life of the bottle do the odds of direct conflict between 
 the children over their choices (i.e., the only purple vitamins are cats or vice-versa) rise above 
 50%? Does it matter which child is allowed to choose first?
 
 Problem by Jean Kahler, 3/30/3016, Analysis by Nathan VC, 4/2016
'''

import string
import itertools as it
import random
import numpy as np
import matplotlib.pyplot as plt

# Generic useful functions
# ---
# Find indices in list of strings that include a specified substring
def findincludes(s, ch):
    return [i for i, ltr in enumerate(s) if ch in str(ltr)]     

# Find indices in list that are exactly equal to a specified element
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

# Generate a bottle of vitamins. The Bottle contains Num_bot vitamins, 
# divided evenly across Num_col colors and Num_shp shapes. The numbers of 
# colors and shapes need to divide evenly into the total number of vitamins. 
# Colors are indicated by lower case letters, and shapes are indicated by 
# numbers. 
def newbottle(Num_bot, Num_col, Num_shp):
    col_lst = list(string.ascii_lowercase)[0:Num_col]
    shp_lst = [str(x) for x in range(0,Num_shp)]
    
    # each entry in AllVits represents a different vitamin, each has equal probability
    Vits = list(it.product(col_lst,shp_lst))

    # Number of each vitamin type (needs to be integer, if not, need to handle differently)
    NumVit = int(Num_bot/len(Vits))

    #List of all items in bottle at start
    NewBottle = Vits*NumVit
    return NewBottle

# Nice child chooses single vitamin from an offered subset of vitamins.
# Child avoids sibling's preferred vitamins if possible.
# Returns offered vitamins with the selected vitamin removed, 
# identity and index of the chosen vitamin, and integer indicating
# whether the child likes the vitamin (happy = 1) or not (happy = 0).
def chooseone_nice(select, select_ind, type_pref, child):  
    
    # identify preferred vitamins for each kid
    pref = []
    for c in range(len(type_pref)):
        pref.append(findincludes(select, type_pref[c]))  
    
    #non preferred vitamins in offered vitamins
    npref = [s for s in range(len(select)) if s not in pref[child]]    
    
    # make list of vitamins preferred by other children
    otherpref = []
    otherkids = list(range(len(type_pref)))
    del(otherkids[child])
    for c in otherkids:
        otherpref = otherpref + pref[c]
    
    # list of preferred vitamins that are not also preferred by other children
    pref_restr=[a for a in pref[child] if a not in otherpref]

    # Make choice
    if pref_restr: # choose first from vits sibling does not prefer
        choice = pref_restr[0]
        happy = 1 # child happy
    elif not pref_restr: 
        if pref[child]: # choose second from own preferences, even if in conflict with sibling
            choice = pref[child][0]
            happy = 1 # child happy
        if not pref[child]:
            choice = npref[0] #choose first non-optimal choice
            happy = 0 # child not happy
    
    # identity of chosen vitamin
    ch_vit = select[choice]
    ch_ind = select_ind[choice] 
    
    # remove chosen vitamin and return
    del(select[choice])
    del(select_ind[choice])           
    
    return(select, select_ind, ch_vit, ch_ind, happy)

# Mean child chooses single vitamin from an offered subset of vitamins.
# Child will choose siblings preferred vitamin if possible, 
# while still respecting their own preference.
# Returns offered vitamins with the selected vitamin removed, 
# the identity and index of the chosen vitamin, and an integer indicating
# whether the child likes the vitamin (happy = 1) or not (happy = 0).
def chooseone_mean(select, select_ind, type_pref, child):  
    
    # identify preferred vitamins for each kid
    pref = []
    for c in range(len(type_pref)):
        pref.append(findincludes(select, type_pref[c]))  
    
    # non-preferred vitamins in offered vitamins
    npref = [s for s in range(len(select)) if s not in pref[child]]    
    
    # make list of vitamins preferred by other children
    otherpref = []
    otherkids = list(range(len(type_pref)))
    del(otherkids[child])
    for c in otherkids:
        otherpref = otherpref + pref[c]
    
    # list of preferred vitamins that ARE also preferred by other children
    pref_restr=[a for a in pref[child] if a in otherpref]
    # list of non-preferred vitamins that other children DO prefer
    npref_restr=[a for a in npref if a in otherpref]
    
    # Make choice
    # first choose from vitamins sibling DOES prefer
    if pref_restr: 
        choice = pref_restr[0]
        happy = 1 # child happy
    # choose second from own preferences that sibling doesn't want
    elif not pref_restr: 
        if pref[child]: 
            choice = pref[child][0]
            happy = 1 # child happy
        # if no preferred available, 
        # pick first from non-preferred that sibling wants
        # and second from the vitamins no one wants
        if not pref[child]:
            if npref_restr:
                choice = npref_restr[0] 
                happy = 0 # child not happy
            elif not npref_restr:
                choice = npref[0] 
                happy = 0 # child not happy
    
    # identity of chosen vitamin
    ch_vit = select[choice]
    ch_ind = select_ind[choice] # index in Bottle vector
    
    # remove chosen vitamin and return
    del(select[choice])
    del(select_ind[choice])           
    
    return(select, select_ind, ch_vit, ch_ind, happy)

# Practical child chooses single vitamin from an offered subset of vitamins.
# If a child has already chosen a preferred vitamin,
# they choose a non-preferred vitamin second. 
# Child also chooses benevolently.
# This function takes an extra entry (happy_ind) and entry of 0 or 1 that indicates 
# whether the child has already chosen one preferred vitamin
# Returns offered vitamins with the selected vitamin removed, 
# identity and index of the chosen vitamin, and integer indicating
# whether the child likes the vitamin (happy = 1) or not (happy = 0)

def chooseone_pract(select, select_ind, type_pref, child, happy_ind):  
    
    # identify preferred vitamins for each kid
    pref = []
    for c in range(len(type_pref)):
        pref.append(findincludes(select, type_pref[c]))  
    
    #non preferred vitamins in offered vitamins
    npref = [s for s in range(len(select)) if s not in pref[child]]    
    
    # make list of vitamins preferred by other children
    otherpref = []
    otherkids = list(range(len(type_pref)))
    del(otherkids[child])
    for c in otherkids:
        otherpref = otherpref + pref[c]
    
    # list of preferred vitamins that are not also preferred by other children
    pref_restr=[a for a in pref[child] if a not in otherpref]

    # Make choice
    if pref_restr and not happy_ind: # choose first from vits sibling does not prefer
        choice = pref_restr[0]
        happy = 1 # child happy
    elif not pref_restr and not happy_ind: 
        if pref[child]: # choose second from own preferences, even if in conflict with sibling
            choice = pref[child][0]
            happy = 1 # child happy
        elif not pref[child]: # choose second from own preferences, even if in conflict with sibling
            choice = npref[0]
            happy = 0 # child happy
    elif happy_ind:
        if npref:
            choice = npref[0] #choose first non-optimal choice
            happy = 0 # child not happy
        if not npref:
            choice = pref[child][0] # choose preferred vit only if no other choice
            happy = 1
    
    # identity of chosen vitamin
    ch_vit = select[choice]
    ch_ind = select_ind[choice] # index in Bottle vector
    
    # remove chosen vitamin and return
    del(select[choice])
    del(select_ind[choice])           
    
    return(select, select_ind, ch_vit, ch_ind, happy)

# Combine single choices into a turn. 
# Bottle is current contents of whole bottle,
# offered is number offered on this turn,
# type_pref is list of preferred vits for each child, 
# kidorder is order of choosing a that constitutes the first turn, 
# nice is string of value 'Nice', 'Mean' or 'Pract'
# Returns revised bottle with selected vitamins removed, 
# identity of chosen vitamins, and a list happy_ro indicating
# whether child choosing at that point liked or did 
# not like the vitamin chosen (both are list of lists organized by child)
def taketurn(Bottle, offered, type_pref, kidorder, nice):

    # indices of selected vitamins
    select_ind = random.sample(range(len(Bottle)), min(offered, len(Bottle)))
    # identity of selected vitamins
    select = [Bottle[s] for s in select_ind]    

    # go through turn order, choosing vitamins sequentially
    happy = [0]*len(kidorder) # will hold whether or not each kid was happy
    ch_vit = [0]*len(kidorder) # will hold vitamins each kid picked
    ch_ind = [0]*len(kidorder) # will hold indices in "Bottle" of vits picked
    happy_ro = [[] for i in range(len(set(kidorder)))] # will hold like/dislike indicator
    ch_vit_ro = [[] for i in range(len(set(kidorder)))] # will hold reordered vit info
    
    # loop through each child in kidorder, choosing single vitamins
    for i, child in enumerate(kidorder):
        if nice == 'Nice':   
            (select, select_ind, ch_vit[i], ch_ind[i], happy[i]) = chooseone_nice(select, select_ind, type_pref, child)
        elif nice == 'Mean':
            (select, select_ind, ch_vit[i], ch_ind[i], happy[i]) = chooseone_mean(select, select_ind, type_pref, child)
        elif nice == 'Pract':
            h=0
            for v in [f for f in find(kidorder, child) if f <= i]:
                h = h + happy[v]
            (select, select_ind, ch_vit[i], ch_ind[i], happy[i]) = chooseone_pract(select, select_ind, type_pref, child, h)
                      
    # reorganize happiness and vitamins by kid
    for c in list(set(kidorder)):
        for v in find(kidorder, c):
            ch_vit_ro[c].append(ch_vit[v])
            happy_ro[c].append(happy[v])
            
    # remove vitamins chosen from the full Bottle
    ch_ind.sort()
    ch_ind.reverse()
    for v in ch_ind:
        del(Bottle[v])
        
    # return new Bottle, the selected vitamins, and marker of whether child was happy
    return(Bottle, ch_vit_ro, happy_ro)

# increment all turns by one in child order
def newturn(orig_turn):
    numkids = len(set(orig_turn))
    return [(i + 1) % numkids for i in orig_turn]

# Simulate turns for an entire bottle
# The first turn taken is defined by turnorder
# nice indicates 'Nice', 'Mean', 'Pract' choices by children
def one_sim(Bottle, Num_off, type_pref, turnorder, nice):
    happy_all = []
    while len(Bottle)>0:
        (Bottle, ch_vit_ro, happy_ro) = taketurn(Bottle, Num_off, type_pref, turnorder, nice)
        happy_all.append(happy_ro) 
        turnorder = newturn(turnorder) # switch who goes first
    happy_sum = np.sum(np.array(happy_all),2)
    return happy_sum

# Generate multuple (numrun) simulated bottles of vitamins
def many_sims(numrun, Num_bot, Num_col, Num_shp, Num_off, type_pref, turnorder, nice):
    happy_sims = np.empty(shape=(numrun, int(Num_bot/len(turnorder)), 
                          len(set(turnorder))), dtype=float)
    for r in range(numrun):
        Bottle = newbottle(Num_bot, Num_col, Num_shp)
        happy_sims[r,:,:] = one_sim(Bottle, Num_off, type_pref, turnorder, nice)
    return happy_sims

# Generate cumulative counts of fits/meltdowns per child over multiple simulations
def cumulative_fits(happy_sims):
    # identify days that child got no preferred vitamins
    fit_matrix = happy_sims == 0
    # cumulative count of child had a giant meltdown 
    # fit due to not getting any preferred vitamins
    fit_cumul = np.cumsum(fit_matrix, 1)
    # take average cumulative count of fits over all simulated vitamin bottles
    fit_cumul_avg = np.mean(fit_cumul, 0)
    # count days that parent has a bad day & add to output
    fit_eitherkid = np.sum(fit_matrix,2) > 0
    badday_cumul = np.cumsum(fit_eitherkid, 1)
    badday_cumul_avg = np.reshape(np.mean(badday_cumul, 0), (len(fit_cumul_avg),1))
    fit_cumul_avg = np.hstack((fit_cumul_avg, badday_cumul_avg))
    return(fit_cumul_avg, fit_cumul)

# Simulated probability of fit for each child on a given day
def prob_fits(happy_sims):
    # identify days that child got no preferred vitamins
    fit_matrix = happy_sims == 0
    # count days that parent has a bad day & add to output
    fit_eitherkid = np.sum(fit_matrix,2) > 0      
    # take mean across simulated vitamin bottles, 
    # this is an approximation of probability of fit on a given day
    fit_prob = np.mean(fit_matrix, 0)
    # calculate probability of bad day for the parent
    fit_prob_parent = np.reshape(np.mean(fit_eitherkid,0), (len(fit_prob),1))
    # add parent to output
    fit_prob = np.hstack((fit_prob, fit_prob_parent))
    return(fit_prob, fit_matrix)

# Generate a simulation and calculate info for probabilities & meltdowns
# Nice takes on value 'Nice', 'Mean' or 'Pract' to define vitamin selection order
def gensim(numrun, Num_bot, Num_col, Num_shp, Num_off, type_pref, turnorder, nice):     
    happy_sims = many_sims(numrun, Num_bot, Num_col, Num_shp, Num_off, type_pref, turnorder, nice)
    (fit_avg, fit_indiv) = cumulative_fits(happy_sims)
    (fit_prob, fit_matrix) = prob_fits(happy_sims)
    return(happy_sims, fit_avg, fit_indiv, fit_prob, fit_matrix)

# Generate a label for preference of each child to use for plots.
# Currently this function is specific to four shapes and three colors
def labelpref(type_pref):
    label_pref = []
    col_labs = ['purple', 'orange', 'pink']
    shp_labs = ['cat', 'lion', 'hippo', 'elephant']
    type_lst = list(string.ascii_lowercase)[0:len(col_labs)] + [str(x) for x in range(0,len(shp_labs))]
    lab_list = col_labs + shp_labs
    child_list = list(string.ascii_uppercase)[0:len(type_pref)]
    for i,t in enumerate(type_pref):
        label_pref.append('Child ' + child_list[i] +': ' + lab_list[find(type_lst, t)[0]])
    return label_pref

# Generate a plot of cumulative count of temper tantrums ("fits") per child
def plot_fitcount(fit_avg, type_pref):
    # plot cumulative count count of temper tantrums    
    plt.plot(fit_avg, marker='.', markersize=10)
    lab_pref = labelpref(type_pref) 
    lab_pref.extend(['Parent'])
    plt.legend(lab_pref, loc=2)
    plt.xlabel('Day', fontsize=20)
    plt.ylabel('Cumulative count of bad days', fontsize=20)
    plt.ylim([0,35])

# Function to generate a plot of fit probability per child
def plot_fitprob(fit_prob, type_pref):
    # fit probability per day (based on simulations)   
    #plt.figure()
    plt.plot(fit_prob, marker='.', markersize=10)
    lab_pref = labelpref(type_pref)
    lab_pref.extend(['Parent'])
    plt.ylim([0,1])
    plt.legend(lab_pref, loc=2)
    plt.xlabel('Day', fontsize=20)
    plt.ylabel('Probability of disappointment', fontsize=20)

# Run simulations and generate plots of outcome.
def runandplot(numrun, Num_bot, Num_col, Num_shp, Num_off, type_pref, turnorder, title_in, nice):
    (happy_sims, fit_avg, fit_indiv, fit_prob, fit_matrix) = \
        gensim(numrun, Num_bot, Num_col, Num_shp, Num_off, type_pref, turnorder, nice)
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)    
    plot_fitcount(fit_avg, type_pref)
    plt.title(title_in, fontsize=20)
    plt.subplot(1,2,2)
    plot_fitprob(fit_prob, type_pref)
    return(happy_sims, fit_avg, fit_indiv, fit_prob, fit_matrix)
