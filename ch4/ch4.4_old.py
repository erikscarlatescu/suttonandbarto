from math import *

import numpy as np

def infinite_sum(lmbd, k, num_terms=10000):
    total = 0

    logsum = 0
    for i in range(1, k):
        logsum += log(i)

    for x in range(num_terms):
        logsum += log(x + k)

        total += exp(-lmbd) * exp(x * log(lmbd) - logsum)

    return total

# this shouldn't work very far, but it seems I've underestimated python
def infinite_sum_unstable(lmbd, k, num_terms=10000):
    total = 0
    for x in range(num_terms):
        total += (lmbd ** x) / factorial(x+k) * exp(-lmbd)
    return total

def finite_sum(lmbd, k):
    total = 0
    for x in range(k):
        total += (lmbd ** x) / factorial(x)

    return exp(-lmbd) * (lmbd ** (-k)) * (exp(lmbd) - total)

def poisson(lmbd, x):
    if x < 0:
        return 0
    return (lmbd ** x) / factorial(x) * exp(-lmbd)

def rent_dist_given_state(lmbd, x, state):
    if x < 0 or x > state:
        return 0

    if x < state:
        return poisson(lmbd, x)

    # probability that all the cars in the store are rented
    total = 0
    for j in range(state):
        total += poisson(lmbd, j)
    return 1 - total

def return_dist_given_everything(lmbd, x, state, num_rents, limit):
    if x >= 0 and x < limit - (state - num_rents):
        return poisson(lmbd, x)

    if x < 0 or x > limit - (state - num_rents):
        return 0

    # probability that so many cars are returned that the lot is full
    total = 0
    for j in range(limit - (state - num_rents)):
        total += poisson(lmbd, j)
    return 1 - total

def transition_prob(limit, rent_lmbd, return_lmbd, state2, action, state1):
    total = 0
    for k in range( (state1-action) +1):
        total += rent_dist_given_state(rent_lmbd, k, (state1-action)) *\
                 return_dist_given_everything(return_lmbd, state2-(state1-action)+k, state1-action, k, limit)
    return total

prob_1_memo = {}
def transition_prob_1_memo(limit, rent_lmbd, return_lmbd, state2, action, state1):
    if (state2, action, state1) in prob_1_memo:
        return prob_1_memo[(state2, action, state1)]

    prob = transition_prob(limit, rent_lmbd, return_lmbd, state2, action, state1)
    prob_1_memo[(state2, action, state1)] = prob
    return prob

prob_2_memo = {}
def transition_prob_2_memo(limit, rent_lmbd, return_lmbd, state2, action, state1):
    if (state2, action, state1) in prob_2_memo:
        return prob_2_memo[(state2, action, state1)]

    prob = transition_prob(limit, rent_lmbd, return_lmbd, state2, action, state1)
    prob_2_memo[(state2, action, state1)] = prob
    return prob

def reward(limit, rent_lmbd, action, state1):
    expected_rentals = 0
    for rent in range(limit):
        expected_rentals += rent * rent_dist_given_state(rent_lmbd, rent, state1-action)
    return 10 * expected_rentals - 2 * action

def expected_rentals(limit, rent_lmbd, action, state1):
    expected_rentals = 0
    for rent in range(limit):
        expected_rentals += rent * rent_dist_given_state(rent_lmbd, rent, state1-action)
    return expected_rentals

def rent_dist_given_state_transition(limit, rent_lmbd, return_lmbd, x, state1, action, state2):
    likelihood = return_dist_given_everything(rent_lmbd, state2 - state1 + action + x, state1, x, limit)
    prior = rent_dist_given_state(rent_lmbd, x, state1 - action)
    normalization = transition_prob(limit, rent_lmbd, return_lmbd, state2, action, state1)

    return likelihood * prior / normalization

def reward_function(limit, rent_lmbd_1, rent_lmbd_2, action, i, j):
    return 10 * (expected_rentals(limit, rent_lmbd_1, action, i) + expected_rentals(limit, rent_lmbd_2, -action, j)) - 2 * abs(action)

# def reward_function(limit, rent_lmbd1, rent_lmbd_2, action, i, j, ni, nj):
#     # if action >= 1: # the employee shuttling a car for free
#     #     action -= 1
#     # return 10 * (expected_rentals(limit, rent_lmbd_1, action, i) + expected_rentals(limit, rent_lmbd_2, -action, j)) - 2 * abs(action)
#
#     pass

if __name__ == "__main__":
    print("Hello world")

    limit = 20

    rent_lmbd_1 = 3
    rent_lmbd_2 = 4

    return_lmbd_1 = 3
    return_lmbd_2 = 2

    policy = np.zeros( (limit, limit) ) # initial policy is to just never move any cars around
    oldvalues = np.zeros( (limit, limit) )

    gamma = 0.9

    # begin policy iteration
    keep_iterating = True
    it = 1
    while keep_iterating:
        print(f"Now beginning iteration {it}")
        # policy evaluation
        keep_evaluating = True
        while keep_evaluating:
            newvalues = np.zeros( (limit, limit) )
            for i in range(limit):
                for j in range(limit):
                    action = round(policy[i, j])
                    reward = reward_function(limit, rent_lmbd_1, rent_lmbd_2, action, i, j)

                    for ni in range(limit):
                        for nj in range(limit):
                            newvalues[i, j] += transition_prob_1_memo(limit, rent_lmbd_1, return_lmbd_1, ni, action, i) *\
                                           transition_prob_2_memo(limit, rent_lmbd_2, return_lmbd_2, nj, -action, j) *\
                                               (reward + gamma * oldvalues[ni, nj])
            delta = np.max(np.abs(newvalues - oldvalues))
            if delta < 0.01:
                keep_evaluating = False
            oldvalues = newvalues
            print(delta)

        # policy improvement
        newpolicy = np.zeros( (limit, limit) )
        for i in range(limit):
            for j in range(limit):
                max_action = -6
                max_val = -1000000
                for action in range(-5, 6):
                    reward = reward_function(limit, rent_lmbd_1, rent_lmbd_2, action, i, j)

                    totalval = 0
                    for ni in range(limit):
                        for nj in range(limit):
                            totalval += transition_prob_1_memo(limit, rent_lmbd_1, return_lmbd_1, ni, action, i) *\
                                           transition_prob_2_memo(limit, rent_lmbd_2, return_lmbd_2, nj, -action, j) *\
                                               (reward + gamma * oldvalues[ni, nj])

                    if totalval > max_val:
                        max_val = totalval
                        max_action = action
                newpolicy[i, j] = max_action

        print(f"{(policy==newpolicy).sum()} differences between the old policy and new policy.")
        if (policy == newpolicy).all():
            keep_iterating = False
        print(newpolicy)
        policy = newpolicy

        it += 1
    np.save("policy", policy)
    np.save("value", oldvalues)

    action = -3
    state1 = 5
    probs = []
    for state2 in range(limit+1):
        probs.append(transition_prob(limit, rent_lmbd_1, return_lmbd_1, state2, action, state1))
    print(probs)
    print(sum(probs))
