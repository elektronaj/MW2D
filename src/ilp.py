'''
Python Multiwinner Package

Using ILP (CPLEX) to solve winner determination for some multiwinner voting rules.
'''


import numpy as np
from itertools import combinations
import numbers
import signal
from argparse import ArgumentParser
import sys
import cplex
from cplex.exceptions import CplexError
import math
from collections import defaultdict
from networkx.algorithms.flow import max_flow_min_cost
import networkx as nx
import heapq
import random
import os


def create_election(n,m,rep=False,seed=None):
    ''' 
    n: number of voters
    m: number of candidates (alternatives)
    each row is a vote (ranked candidates in total order)
    '''
    assert type(m) in [list,int], "expected list of candidates or int"
    
    np.random.RandomState(seed)
    election = np.tile(np.arange(m) if type(m) is int else m,(n,1))

    no_of_candidates = m if type(m) is int else len(m)
    
    combinations = math.factorial(no_of_candidates)

    if not rep:
        map(np.random.shuffle, election)
        return election
    else:
        rep = math.factorial(m) * 0.05 - 1
        idx = np.zeros((n),dtype=np.int)
        seen = 0
        for i in range(n):
            p = np.random.uniform()
            if p < combinations / float(combinations + seen):
                np.random.shuffle(election[i,:])
                idx[i] = i
            else:
                idx[i] = idx[np.random.randint(i)]
            seen += rep

    return election[idx,:]


def brute_force_CC(election,k,scoring=None):
    ''' 
    Checks all n choose k committees and choose the best one,
    acording to the represantive value
    '''
    n,m = election.shape
    candidates = np.sort(election[0,:])

    if scoring is not None:
        assert m == len(scoring), "mismatch: number of candidates and number of scores"
    else:
        scoring = np.arange(m-1,-1,-1) # = np.tile(np.arange(m-1,-1,-1) if scoring is None else scoring,(n,1))
    
    best_score = float("-inf")
    best_committee = None
    
    tmp_committee = np.zeros((m))

    for committee in combinations(candidates,k):
        tmp_election = np.zeros(election.shape)

        for candidate in committee:
            tmp_election += election == candidate

        tmp_sum = np.sum(scoring[np.argmax(tmp_election,axis=1)])
        
        if tmp_sum > best_score:
            best_score = tmp_sum
            best_committee = committee
                
    return best_score, best_committee


def k_borda(election,k,scoring=None):
    ''' 
    Currently ignoring the order if scoring doesn't break ties,
    see scoring with all ones.
    '''
    n,m = election.shape
    if scoring is not None:
        assert m == len(scoring), "mismatch: number of candidates and number of scores"
    else: 
        scoring = np.arange(m-1,-1,-1) # np.tile(np.arange(m-1,-1,-1) if scoring is None else scoring,(n,1))
    
    candidates = np.sort(election[0,:])
    
    candidate_score = np.zeros((m))
    for i,candidate in enumerate(candidates):
        candidate_score[i] = np.sum((election == candidate) * scoring)

    return candidates[np.argsort(candidate_score)[::-1]][:k]


def distance(vote,candidate):
    return np.argmax(vote == candidate)


def clustering(election,k,committee=None,scoring=None,iterations=30,restart=True,fill_greedy=False,init_alpha=False,alpha=1):
    ''' 
    Currently ignoring the order if scoring doesn't break ties,
    see scoring with all ones.
    '''
    n,m = election.shape
    if scoring is not None:
        assert m == len(scoring), "mismatch: number of candidates and number of scores"
    else: 
        scoring = np.arange(m-1,-1,-1)
    # ^^^ factor out
    candidates = np.sort(election[0,:])
    
    if committee is not None:
        assert len(committee) == k, "committee size has to match k"
    else:
        if init_alpha:
            committee = alpha_committee(election,k,alpha)
        else:
            committee = np.random.choice(candidates,k,replace=False)
    
    last_committee = None
    
    best_committee = np.copy(committee)
    best_score = score_committee(election,committee,scoring)
    
    iterations_since_convergence = 0
    
    restarts = []
    
    for it in range(iterations):
        in_cluster = np.zeros((n))
        # check for convergence
        if last_committee is not None and np.array_equal(committee,last_committee):
        # restart! if it < iterations, and store best solution
            score_ = score_committee(election,committee,scoring)
            if restart:
                if best_committee is None or score_ > best_score:
                    best_committee = committee
                    best_score = score_
                    
                restarts.append(iterations_since_convergence)
                iterations_since_convergence = 0
                if init_alpha:
                    committee = alpha_committee(election,k,alpha)
                else:
                    committee = np.random.choice(candidates,k,replace=False)
            else:
                return committee, score_, iterations_since_convergence
        
        last_committee = np.copy(committee)
        
        for i in range(n):
            positions = map(lambda c: distance(election[i],c), committee)
            in_cluster[i] = np.argmin(positions)
            
        for c_idx in range(k):
            votes = election[in_cluster == c_idx,:]
            if votes.shape[0] == 0:
                # if the 'perfect' solution is < k, other candidates arbitrary?
                continue
                
            new_c = k_borda(votes,1)
            committee[c_idx] = new_c[0]
            
        current_candidates = np.unique(committee)
        unique_candidates = current_candidates.shape[0]
        
        if unique_candidates < k:
            rest = k - unique_candidates
            committee[:unique_candidates] = current_candidates
            remaining_candidates = np.setdiff1d(candidates,current_candidates)

            if fill_greedy:
                committee[unique_candidates:] = greedyCC(election,rest,remaining_candidates)
            else:
                committee[unique_candidates:] = np.random.choice(remaining_candidates,rest,replace=False)
        
        iterations_since_convergence += 1
    
    return (best_committee,best_score,np.mean(restarts)) if best_committee is not None else (committee,score_committee(election,committee,scoring),iterations_since_convergence)


def add_to_pq(pq,element,threshold,size):
    heapq.heappush(pq,element)
    if size == threshold:
        heapq.heappop(pq)
    return pq[0][0]


def score_committee(election,committee,scoring=None):
    n,m = election.shape

    if scoring is not None:
        assert m == len(scoring), "mismatch: number of candidates and number of scores"
    else:
        scoring = np.arange(m-1,-1,-1)
    
    assert len(committee) > 0, "empty committee"
    
    res = np.argmax(election == committee[0],axis=1)
    
    for candidate in committee[1:]:
        res = np.minimum(res,np.argmax(election == candidate,axis=1))

    return np.sum(scoring[res])


def get_clusters(fd,n,k):
    in_cluster = defaultdict(list)

    for i in range(n):
        inserted = False
        for (k,v) in fd[i].iteritems():
            if v == 1:
                in_cluster[k].append(i)
                inserted = True
                break
        if not inserted:
            in_cluster['left_over'].append(i)

    return in_cluster
    

def clustering_monroe(election,k,committee=None,committee_score=0,scoring=None,iterations=30, restart=False):
    n,m = election.shape
    if scoring is not None:
        assert m == len(scoring), "mismatch: number of candidates and number of scores"
    else: 
        scoring = np.arange(m-1,-1,-1) # need to be reversed for max flow min cost alg.
    # ^^^ factor out
    candidates = np.sort(election[0,:])
    
    if committee is not None:
        assert len(committee) == k, "committee size has to match k"
    else:
        committee = np.random.choice(candidates,k,replace=False)
    
    last_committee = None
    
    best_committee = np.copy(committee)
    best_score = committee_score
    
    iterations_since_convergence = 0
    
    restarts = []
    
    d = {}
    d_rev = {}
    d_ = {}
    d_rev_ = {}
    for i,c in enumerate(candidates):
        d[c] = i + n
        d_rev[i+n] = c
        d_[c] = i + k
        d_rev_[i+k] = c
    
    s = n + m
    t = s + 1
    s_ = k + m
    t_ = s_ + 1
    
    ''' first n nodes are the voters, next m nodes are the candidates last to are s,t '''
            
    last_committee = None
    
    iterations_since_convergence = 0
    
    restarts = []
    
    scoring_rev = np.max(scoring) - scoring
    
    for it in range(iterations):
        if last_committee is not None and np.array_equal(committee,last_committee):
        # restart! if it < iterations, and store best solution
            #score_ = score_committee(election,committee) #,scoring)
            if restart:
                if best_committee is None or score_ > best_score:
                    best_committee = committee
                    best_score = score_
                    
                restarts.append(iterations_since_convergence)
                iterations_since_convergence = 0
                committee = np.random.choice(candidates,k,replace=False)
            else:
                return committee, score_, iterations_since_convergence
        
        last_committee = np.copy(committee)
        
        G = nx.DiGraph()
        for v in range(n):
            G.add_edge(s,v,{'capacity': 1, 'weight': 1})
            for c in committee:
                G.add_edge(v,d[c],{'capacity': 1, 'weight': scoring_rev[np.argmax(election[v,:] == c)]})
                
        for c in committee:
            G.add_edge(d[c],t,{'capacity': n / k, 'weight': 1})

        fd = max_flow_min_cost(G,s,t)
        
        cl = get_clusters(fd,n,k)
        
        if n % k != 0:
            # flow to distribute the left over voters
            G_rest = nx.DiGraph()
            
            for v in cl['left_over']:
                G_rest.add_edge(s,v,{'capacity': 1, 'weight': 1})
                for c in committee:
                    G_rest.add_edge(v,d[c],{'capacity': 1, 'weight': scoring_rev[np.argmax(election[v,:] == c)]})
                
            for c in committee:
                G_rest.add_edge(d[c],t,{'capacity': 1, 'weight': 1})

            fd_res = max_flow_min_cost(G_rest,s,t)
                        
            for v in cl['left_over']:
                for (c,is_set) in fd_res[v].iteritems():
                    if is_set == 1:
                        cl[c].append(v)
                        
            cl.pop('left_over',None)
            
        G2 = nx.DiGraph()

        committee = []
        cl_map = {}
        for i,nodes in enumerate(cl.itervalues()):
            if i == 'left_over':
                continue
            cl_map[i] = nodes
            for candidate in candidates:
                candidate_score = np.sum((election[nodes,:] == candidate) * scoring_rev)
                G2.add_edge(i,d_[candidate],{'capacity': 1, 'weight': candidate_score})
                
            G2.add_edge(s_,i,{'capacity': 1, 'weight': 1})
        
        for c in candidates:
            G2.add_edge(d_[c],t_,{'capacity': 1, 'weight': 1})
        
        fd2 = max_flow_min_cost(G2,s_,t_)
                
        score_ = 0
        found = 0
        for c in range(k):
            for (candidate,is_set) in fd2[c].iteritems():
                if is_set == 1:
                    candidate_ = d_rev_[candidate]
                    committee.append(candidate_)
                    score_ += np.sum((election[cl_map[c],:] == candidate_) * scoring)
                
        committee = np.array(committee)
        
        iterations_since_convergence += 1
        
    return (best_committee,best_score,0) if best_committee is not None else (committee,score_,iterations_since_convergence)        


def write_cplex_format(election,k,out,scoring=None):
    n,m = election.shape
    
    if scoring is not None:
        assert m == len(scoring), "mismatch: number of candidates and number of scores"
    else: 
        scoring = np.arange(m-1,-1,-1)
    
    candidates = np.sort(election[0,:])
    
    d = {}
    
    for i,c in enumerate(candidates):
        d[c] = "x" + str(i) 
    
    f = open(out,'w')
    s = "Maximize\nobj:"
    
    subj_k = ""
    pos = 0
    first = True
    for i in range(n):
        for j in range(m):
            if not first:
                s += " +"
            first = False
            s += " " + str(scoring[j]) + " y" + str(pos)
            
            pos += 1
            
    f.write(s+"\n")
    f.write("Subject To\n")
    subj_k = "c1:"
    first = True
    for c in candidates:
        if not first:
            subj_k += " +"
        first = False
        subj_k += " " + d[c]
        
    f.write(subj_k+' = ' + str(k) + '\n')
    
    pos = 0
    yli = ""
    for i in range(n):
        #c = " c" + str(i+2) + ":"
        c = "c" + str(i+2) + ": "
        first = True
        for c_ in election[i,:]:
            if not first:
                c += " + "
            first = False
            c += "y" + str(pos)
            yli += "c" + str(pos+n+2)+ ": y" + str(pos) + " - " + d[c_] + " <= 0\n"
            pos += 1
        c += " = 1\n"
        f.write(c)
    f.write(yli)
            
    # k
    f.write("Binary\n")
    for i in range(n*m):
        f.write("y"+str(i) + "\n")
        if i < m:
            f.write("x"+str(i) +"\n")
        
    f.write("End\n")


def write_cplex_format_pav(election,k,out,scoring=None):
    n,m = election.shape
    
    if scoring is not None:
        assert m == len(scoring), "mismatch: number of candidates and number of scores"
    else: 
        scoring = np.arange(m-1,-1,-1)
    
    candidates = np.sort(election[0,:])
    
    d = {}
    
    for i,c in enumerate(candidates):
        d[c] = "x" + str(i) 
    
    f = open(out,'w')
    s = "Maximize\nobj:"
    
    subj_k = ""
    pos = 0
    first = True
    for z in range(k):
        for i in range(n):
            for j in range(m):
                if not first:
                    s += " +"
                first = False
                s += " " + str(scoring[j] * 1. / (z + 1)) + " " + "y" + str(pos)
                pos += 1
            
    f.write(s+"\n")
    f.write("Subject To\n")

    # constraint for x's
    subj_k = "c1:"
    first = True
    for c in candidates:
        if not first:
            subj_k += " +"
        first = False
        subj_k += " " + d[c]
        
    f.write(subj_k+' = ' + str(k) + '\n')

    # constraints for y's    
    pos = 0
    yli = ""
    for z in range(k):
        for i in range(n):
            #c = " c" + str(i+2) + ":"
            c = "c" + str(i+2+(n*z)) + ": "
            first = True
            for c_ in election[i,:]:
                if not first:
                    c += " + "
                first = False
                c += "y" + str(pos)
                yli += "c" + str(pos+n+2+(n*(k - 1)))+ ": y" + str(pos) + " - " + d[c_] + " <= 0\n"
                pos += 1
            c += " = 1\n"
            f.write(c)
    f.write(yli)
            
    # new constraints for pav
    for j in range(m):
        for i in range(n):
            s = 'c' + str(n+2+(n*(k - 1)) + n*m*k + i + n*j) + ":"
            first = True
            for z in range(k):
                if not first:
                    s += " +"
                s += " " + "y" + str(i + n*j + n*m*(z))
                first = False
            s += ' <= 1'
            s += '\n'
            f.write(s)

    # k
    f.write("Binary\n")
    for i in range(n*m*k):
        f.write("y"+str(i) + "\n")
        if i < m:
            f.write("x"+str(i) +"\n")
        
    f.write("End\n")
    

def write_cplex_format_pavtopk(election,k,out,scoring=None):
    n,m = election.shape
    
    if scoring is not None:
        assert m == len(scoring), "mismatch: number of candidates and number of scores"
    else: 
        scoring = np.array([1] * k + [0] * (m - k))
    
    candidates = np.sort(election[0,:])
    
    d = {}
    
    for i,c in enumerate(candidates):
        d[c] = "x" + str(i) 
    
    f = open(out,'w')
    s = "Maximize\nobj:"
    
    subj_k = ""
    pos = 0
    first = True
    for z in range(k):
        for i in range(n):
            for j in range(m):
                if not first:
                    s += " +"
                first = False
                s += " " + str(scoring[j] * 1. / (z + 1)) + " " + "y" + str(pos)
                pos += 1
            
    f.write(s+"\n")
    f.write("Subject To\n")

    # constraint for x's
    subj_k = "c1:"
    first = True
    for c in candidates:
        if not first:
            subj_k += " +"
        first = False
        subj_k += " " + d[c]
        
    f.write(subj_k+' = ' + str(k) + '\n')

    # constraints for y's    
    pos = 0
    yli = ""
    for z in range(k):
        for i in range(n):
            #c = " c" + str(i+2) + ":"
            c = "c" + str(i+2+(n*z)) + ": "
            first = True
            for c_ in election[i,:]:
                if not first:
                    c += " + "
                first = False
                c += "y" + str(pos)
                yli += "c" + str(pos+n+2+(n*(k - 1)))+ ": y" + str(pos) + " - " + d[c_] + " <= 0\n"
                pos += 1
            c += " = 1\n"
            f.write(c)
    f.write(yli)
            
    # new constraints for pav
    for j in range(m):
        for i in range(n):
            s = 'c' + str(n+2+(n*(k - 1)) + n*m*k + i + n*j) + ":"
            first = True
            for z in range(k):
                if not first:
                    s += " +"
                s += " " + "y" + str(i + n*j + n*m*(z))
                first = False
            s += ' <= 1'
            s += '\n'
            f.write(s)

    # k
    f.write("Binary\n")
    for i in range(n*m*k):
        f.write("y"+str(i) + "\n")
        if i < m:
            f.write("x"+str(i) +"\n")
        
    f.write("End\n")


def write_cplex_format_monroe(election,k,out,scoring=None):
    n,m = election.shape
    scoring = scoring if scoring is not None else range(m-1,-1,-1)

    # scoring = np.max(scoring) - scoring
    
    candidates = np.sort(election[0,:])
    
    d = {}
    c_map = defaultdict(list)
    
    for i,c in enumerate(candidates):
        x = 'x' + str(i)
        d[c] = x
    
    f = open(out,'w')
    
    obj = 'Maximize\nobj: '
    
    var = 'Binary\n'
        
    obj_list = []
    c2 = ''
    c6 = ''    
    
    for i in range(n):
        c2_list = []
        for j,cand in enumerate(election[i,:]):
            z = 'z' + str(i) + '_' + str(j)
            var += z + '\n'
            obj_list.append(str(scoring[j]) + ' ' + z)
            c2_list.append(z)
            c6 += 'c6_' + str(i) + '_' + str(j) + ': ' + z + ' <= 1\n'
            c_map[d[cand]].append(z) 
        c2 += 'c2_' + str(i) + ': ' + ' + '.join(c2_list) + ' = 1\n'
            
    obj = obj + ' + '.join(obj_list) + '\n'
    f.write(obj)
    
    c1_list = []
    c5 = ''
    for i in range(m):
        x = 'x' + str(i)
        var += x + '\n'
        c1_list.append(x)
        c5 += 'c5_' + str(i) +': ' + x +  ' <= 1 \n'
    
    f.write('Subject To\nc1: ' + ' + '.join(c1_list) + ' = ' + str(k) + '\n')    
    
    L = math.floor(n/float(k))
    U = math.ceil(n/float(k))
    
    c3 = ''
    c4 = ''
    
    for i,(k,v) in enumerate(c_map.iteritems()):
        x = 'x' + str(i)
        xlist = ' + '.join(c_map[x])
        c3 += 'c3_' + str(i) + ': -' + str(L) + ' ' + x + ' + ' + xlist + ' >= 0\n'
        c4 += 'c4_' + str(i) + ': -' + str(U) + ' ' + x + ' + ' + xlist + ' <= 0\n'    
    
    f.write(c2)
    f.write(c3)
    f.write(c4)
    f.write(c5)
    f.write(c6)
    f.write(var)

    f.write("End\n")

    f.close()
    
    
def run_ilp(election,k,scoring=None):
    n,m = election.shape
    
    candidates = np.sort(election[0,:])

    tmp = "tmp_" + str(random.randint(0,3283393292303320932)) + ".lp"

    write_cplex_format(election,k,tmp,scoring)
    cpx = cplex.Cplex(tmp)

    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.set_results_stream(None)

    try:
        cpx.solve()
    except CplexError, exc:
        print exc
        return

    os.remove(tmp)

    x = np.array(cpx.solution.get_values()[-m:])
    
    # return cpx.solution.get_objective_value(), candidates[x == 1.0]
    return cpx.solution.get_objective_value(), candidates[np.logical_and(x > 0.9, x < 1.1)]


def run_ilp_pav(election,k,scoring=None):
    n,m = election.shape
    
    candidates = np.sort(election[0,:])

    tmp = "tmp2_" + str(random.randint(0,3283393292303320932)) + ".lp"

    write_cplex_format_pav(election,k,tmp,scoring)

    cpx = cplex.Cplex(tmp)
    
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.set_results_stream(None)

    try:
        cpx.solve()
    except CplexError, exc:
        print exc
        return

    os.remove(tmp)

    x = np.array(cpx.solution.get_values()[-m:])
    
    # return cpx.solution.get_objective_value(), candidates[x == 1.0]
    return cpx.solution.get_objective_value(), candidates[np.logical_and(x > 0.9, x < 1.1)]


def run_ilp_pavtopk(election,k,scoring=None):
    n,m = election.shape
    
    candidates = np.sort(election[0,:])

    tmp = "tmp2_" + str(random.randint(0,3283393292303320932)) + ".lp"

    write_cplex_format_pavtopk(election,k,tmp,scoring)

    cpx = cplex.Cplex(tmp)
    
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.set_results_stream(None)

    try:
        cpx.solve()
    except CplexError, exc:
        print exc
        return

    os.remove(tmp)

    x = np.array(cpx.solution.get_values()[-m:])
    
    # Returns cpx.solution.get_objective_value(), candidates[x == 1.0]
    return cpx.solution.get_objective_value(), candidates[np.logical_and(x > 0.9, x < 1.1)]


def run_ilp_monroe(election,k,scoring=None):
    n,m = election.shape
    
    candidates = np.sort(election[0,:])

    tmp = "tmp2_" + str(random.randint(0,3283393292303320932)) + ".lp"

    write_cplex_format_monroe(election,k,tmp,scoring)
    cpx = cplex.Cplex(tmp)
    
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.set_results_stream(None)

    try:
        cpx.solve()
    except CplexError, exc:
        print exc
        return

    os.remove(tmp)

    x = np.array(cpx.solution.get_values()[-m:])
    
    # Returns cpx.solution.get_objective_value(), candidates[x == 1.0]
    return cpx.solution.get_objective_value(), candidates[np.logical_and(x > 0.9, x < 1.1)]


def run_greedyCC_noc(election,k,d,p=0.75,e=1.0,scoring=None):
    best, committees = greedyCC_d_p(election,k,d,p=p,e=e,scoring=scoring)                                                                                            
    score = score_committee(election,best,scoring=scoring)                                                                                                                                                                                                                                                   
    _,bs,_ = run_clustering(election,k,committees,scoring=scoring)

    return score,bs


def run_greedy_monroe_noc(election,k,d,p=0.75,e=1.0,scoring=None):
    _, score, committees, scores_ = greedy_monroe_d_p(election,k,d,p=p,e=e,scoring=scoring)                                                                                            
    _,bs,_ = run_clustering_monroe(election,k,committees,scores_,scoring=scoring)

    return score,bs


def super_geil_algorithm_no_clustering(election,k,p=0.0,n_jobs=-1,scoring=None):    
    best_committee = None
    best_score = 0
    best_score_nc = 0
    greedy_scores = []
    greedy_scores_cl = []

    res = []
    scores = []
    
    for d in [1,5]: #[1,5,10,15,20]:
        print d
        best, committees = greedyCC_d_p(election,k,d,p=0,scoring=scoring) # p = 0 equals greedy only
        score = score_committee(election,best,scoring=scoring)
        if score > best_score_nc:
            best_score_nc = score

        greedy_scores.append(score)

        scores.append(score)
        bc,bs,ba = run_clustering(election,k,committees,scoring=scoring)
        scores.append(bs)
        greedy_scores_cl.append(bs)

        res += Parallel(n_jobs=n_jobs)(delayed(run_greedyCC_noc)(election,k,d,scoring=scoring) for _ in range(5))

    for (s,cs) in res:
        scores.append(cs)
        if s > best_score_nc:
            best_score_nc = s

    scores = np.array(scores)

    bs = np.max(scores)
    same = np.sum(scores == bs)
    return greedy_scores,greedy_scores_cl,bs,same,best_score_nc


def super_geil_algorithm_no_clustering_monroe(election,k,p=0.5,n_jobs=-1,scoring=None):    
    best_committee = None
    best_score = 0
    best_score_nc = 0
    greedy_scores = []
    greedy_scores_cl = []

    res = []
    scores = []
    
    for d in [1,5]: #[1,5,10,15,20]:
        print d

        _, score, committees,scores_ = greedy_monroe_d_p(election,k,d,p=0,scoring=scoring)

        if score > best_score_nc:
            best_score_nc = score

        greedy_scores.append(score)

        scores.append(score)
        bc,bs,ba = run_clustering_monroe(election,k,committees,scores_,scoring=scoring)
        scores.append(bs)
        greedy_scores_cl.append(bs)

        res += Parallel(n_jobs=n_jobs)(delayed(run_greedy_monroe_noc)(election,k,d,scoring=scoring) for _ in range(5))

    for (s,cs) in res:
        scores.append(cs)
        if s > best_score_nc:
            best_score_nc = s

    scores = np.array(scores)
    bs = np.max(scores)
    same = np.sum(scores == bs)
    return greedy_scores,greedy_scores_cl,bs,same,best_score_nc


def run_clustering(election,k,committees,scoring=None):
    best_committee =  None
    best_score = 0
    
    committees_ = []
    scores = []
    
    best = ''

    for c in committees:
        committee, score, _ = clustering(election,k,committee=c,scoring=scoring,restart=False)
        committees_.append(committee)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_committee = np.copy(committee)
            best = 'clustering w/ '

    return best_committee, best_score, best


def run_clustering_monroe(election,k,committees,scores_,scoring=None):
    best_committee =  None
    best_score = 0
    
    committees_ = []
    scores = []
    
    best = ''
    for i,c in enumerate(committees):
        committee, score, _ = clustering_monroe(election,k,committee=c,committee_score=scores_[i],scoring=scoring,restart=False)
        committees_.append(committee)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_committee = np.copy(committee)
            best = 'clustering w/ '

    return best_committee, best_score, best


def greedyCC_d_p(election, k, d, switch = None, p = 1.0, e = 1.0, candidates=None,scoring=None):
    """
        p == 1.0 -> greedy
        p == 0.0 -> random
    """
    n,m = election.shape
    
    assert k <= m, "k has to be smaller than the number of candidates"
    
    scoring = scoring if scoring is not None else np.arange(m-1,-1,-1)

    if candidates is not None:
        cc = len(candidates)
        assert cc >= k and cc <= m, "number of candidates has to be at least k and at most m"
    else:
        candidates = np.sort(election[0,:])
        
    candidate_set = set(candidates)
    
    partial_committee = []
    
    # initialize
    partial_committee.append(([],candidate_set))
    
    for i in range(k):
        tmp_partial = []
        p_ = 1 - p ** ((i+1) * e)
        size = 0
        lowest = n * (m-1)
        if p_ == 1 or random.random() <= p_: # avoid calls to random.random() if we aren't in coin-mode
            for (pc,rest) in partial_committee:
                for alternative in rest:
                    pc_ = pc[:]
                    pc_.append(alternative)
                    # somehow avoid all the copying?! can use the alternative to append

                    score = score_committee(election,pc_)
                    if size < d or (size == d and score > lowest):
                        lowest = add_to_pq(tmp_partial,(score,(pc_,rest,alternative)),d,size)

                    if size < d:
                        size += 1

        else:
            d_ = len(partial_committee)
            for _ in range(d):
                pick = np.random.randint(0,d_)
                pc, rest = partial_committee[pick]
                alternative = np.random.choice(list(rest))
                pc_ = pc[:]
                pc_.append(alternative)
                # somehow avoid all the copying?! can use the alternative to append

                score = score_committee(election,pc_)
                if size < d or (size == d and score > lowest):
                    lowest = add_to_pq(tmp_partial,(score,(pc_,rest,alternative)),d,size)

                if size < d:
                    size += 1
        
        partial_committee = []
        for (score,(pc,rest,a)) in tmp_partial: # deterministisch?
            r = rest.copy()
            r.remove(a)
            partial_committee.append((pc[:],r))
    committees = map(lambda x: x[0],partial_committee)

    return partial_committee[-1][0], committees


def greedy_monroe_d_p(election,k,d,committee=None,scoring=None,p=0.0,e=1.0):
    n,m = election.shape
    if scoring is not None:
        assert m == len(scoring), "mismatch: number of candidates and number of scores"
    else: 
        scoring = np.arange(m-1,-1,-1) # need to be reversed for max flow min cost alg.
    # ^^^ factor out
    candidates = np.sort(election[0,:])
    
    if committee is not None:
        assert len(committee) == k, "committee size has to match k"
    else:
        committee = np.random.choice(candidates,k,replace=False)

    dm = {}
    dm_rev = {}
    for i,c in enumerate(candidates):
        dm[c] = i + n
        dm_rev[i+n] = c
    
    s = n + m
    t = s + 1
    
    ''' first n nodes are the voters, next m nodes are the candidates last to are s,t '''
    
    candidates = list(candidates)
    committee = []
        
    no_of_candidates = len(candidates)
    
    idx_left = np.ones((n),dtype=bool)
    
    n2 = n % k
    
    candidate_set = set(candidates)
    
    partial_committee = []
    
    # initialize
    partial_committee.append(([],candidate_set,idx_left))
    
    for i in range(k):
        n_over_k = math.ceil(n/float(k)) if i < n2 else math.floor(n/float(k))
        tmp_partial = []
        p_ = 1 - p ** ((i+1) * e)
        size = 0
        lowest = n * (m-1)
        if p_ == 1 or random.random() <= p_:
            for (pc,rest,idx_left) in partial_committee:
                for candidate in rest:
                    pc_ = pc[:]
                    pc_.append(candidate)
                    # somehow avoid all the copying?! can use the alternative to append

                    candidate_votes = np.max((election[idx_left] == candidate) * scoring,axis=1)
                    votes_idx = np.argsort(candidate_votes)[::-1][:n_over_k]
                    candidate_score = np.sum(candidate_votes[votes_idx])
    
                    if size < d or (size == d and candidate_score > lowest):
                        lowest = add_to_pq_hack(tmp_partial,(candidate_score,(pc_,rest,candidate,idx_left,votes_idx)),d,size)

                    if size < d:
                        size += 1
                        
        else:
            d_ = len(partial_committee)

            for _ in range(d):
                pick = np.random.randint(0,d_)
                pc, rest, idx_left = partial_committee[pick]
                candidate = np.random.choice(list(rest))
                pc_ = pc[:]
                pc_.append(candidate)
                # somehow avoid all the copying?! can use the alternative to append

                candidate_votes = np.max((election[idx_left] == candidate) * scoring,axis=1)
                votes_idx = np.argsort(candidate_votes)[::-1][:n_over_k]
                candidate_score = np.sum(candidate_votes[votes_idx])
        
                if size < d or (size == d and candidate_score > lowest):
                    lowest = add_to_pq_hack(tmp_partial,(candidate_score,(pc_,rest,candidate,idx_left,votes_idx)),d,size)

                if size < d:
                    size += 1
        
        partial_committee = []
        
        for (score,(pc,rest,a,idx_left,votes_idx)) in tmp_partial:
            r = rest.copy()
            r.remove(a)
            idx_left_ = np.copy(idx_left)
            idx_left_[votes_idx] = False
            partial_committee.append((pc[:],r,idx_left_))

    scoring_rev = np.max(scoring) - scoring
    
    candidates = np.sort(election[0,:])
    
    best_score = 0
    best_committee = None
    best_cl = None
    
    scores_ = []
    
    for (committee,_,_) in partial_committee:
        G = nx.DiGraph()
        for v in range(n):
            G.add_edge(s,v,{'capacity': 1, 'weight': 1})
            for c in committee:
                G.add_edge(v,dm[c],{'capacity': 1, 'weight': scoring_rev[np.argmax(election[v,:] == c)]})

        for c in committee:
            G.add_edge(dm[c],t,{'capacity': n / k, 'weight': 1})
            
        fd = max_flow_min_cost(G,s,t)
        cl = get_clusters(fd,n,k)    

        if n % k != 0:
            # flow to distribute the left over voters
            G_rest = nx.DiGraph()
            
            for v in cl['left_over']:
                G_rest.add_edge(s,v,{'capacity': 1, 'weight': 1})
                for c in committee:
                    G_rest.add_edge(v,dm[c],{'capacity': 1, 'weight': scoring_rev[np.argmax(election[v,:] == c)]})
                
            for c in committee:
                G_rest.add_edge(dm[c],t,{'capacity': 1, 'weight': 1})

            fd_res = max_flow_min_cost(G_rest,s,t)
                        
            for v in cl['left_over']:
                for (c,is_set) in fd_res[v].iteritems():
                    if is_set == 1:
                        cl[c].append(v)
                        
            cl.pop('left_over',None)
        
        score = 0

        for candidate in cl.iterkeys():
            score += np.sum((election[cl[candidate],:] == dm_rev[candidate]) * scoring)
        
        if score > best_score:
            best_score = score
            best_committee = np.copy(committee)
            best_cl = cl.copy()

        scores_.append(score)
    
    committees = map(lambda x: x[0],partial_committee)
    
    return best_committee, best_score, committees, scores_


def add_to_pq_hack(pq,element,threshold,size):
    try: # this is a hack: catches duplicate elements in the queue
        heapq.heappush(pq,element)
        if size == threshold:
            heapq.heappop(pq)
    except ValueError:
        pass
    finally:
        return pq[0][0]


def write_cplex_format_OWA(election,k,OWA,out,scoring=None):
    n,m = election.shape
    
    if scoring is not None:
        assert m == len(scoring), "mismatch: number of candidates and number of scores"
    else: 
        scoring = np.arange(m-1,-1,-1)
    
    candidates = np.sort(election[0,:])
    
    d = {}
    
    for i,c in enumerate(candidates):
        d[c] = "x" + str(i) 
    
    f = open(out,'w')
    s = "Maximize\nobj:"
    
    subj_k = ""
    pos = 0
    first = True
    for z in range(k):
        for i in range(n):
            for j in range(m):
                if not first:
                    s += " +"
                first = False
                s += " " + str(scoring[j] * OWA[z]) + " " + "y" + str(pos)
                pos += 1


            
    f.write(s+"\n")
    f.write("Subject To\n")

    # constraint for x's
    subj_k = "c1:"
    first = True
    for c in candidates:
        if not first:
            subj_k += " +"
        first = False
        subj_k += " " + d[c]
        
    f.write(subj_k+' = ' + str(k) + '\n')

    # constraints for y's    
    pos = 0
    yli = ""
    for z in range(k):
        for i in range(n):
            #c = " c" + str(i+2) + ":"
            c = "c" + str(i+2+(n*z)) + ": "
            first = True
            for c_ in election[i,:]:
                if not first:
                    c += " + "
                first = False
                c += "y" + str(pos)
                yli += "c" + str(pos+n+2+(n*(k - 1)))+ ": y" + str(pos) + " - " + d[c_] + " <= 0\n"
                pos += 1
            c += " = 1\n"
            f.write(c)
    f.write(yli)
            
    # new constraints for pav (not pav anymore)
    for j in range(m):
        for i in range(n):
            s = 'c' + str(n+2+(n*(k - 1)) + n*m*k + i + n*j) + ":"
            first = True
            for z in range(k):
                if not first:
                    s += " +"
                s += " " + "y" + str(i + n*j + n*m*(z))
                first = False
            s += ' <= 1'
            s += '\n'
            f.write(s)

    # k
    f.write("Binary\n")
    for i in range(n*m*k):
        f.write("y"+str(i) + "\n")
        if i < m:
            f.write("x"+str(i) +"\n")
        
    f.write("End\n")


def run_ilp_OWA(election,k,OWA,scoring=None):
    n,m = election.shape
    
    candidates = np.sort(election[0,:])

    tmp = "tmp2_" + str(random.randint(0,3283393292303320932)) + ".lp"

    write_cplex_format_OWA(election,k,OWA,tmp,scoring)

    cpx = cplex.Cplex(tmp)
    
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.set_results_stream(None)

    try:
        cpx.solve()
    except CplexError, exc:
        print exc
        return

    os.remove(tmp)

    x = np.array(cpx.solution.get_values()[-m:])
    
    # Returns cpx.solution.get_objective_value(), candidates[x == 1.0]
    return cpx.solution.get_objective_value(), candidates[np.logical_and(x > 0.9, x < 1.1)]
