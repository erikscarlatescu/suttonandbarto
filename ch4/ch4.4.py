from math import *
import numpy as np

def poisson(lmbd, x):
    if x < 0:
        return 0
    return (lmbd ** x) / factorial(x) * exp(-lmbd)

class CarLot:
    def __init__(self, rent_lambda, return_lambda, max_cars):
        self.rent_lambda = rent_lambda
        self.return_lambda = return_lambda
        self.max_cars = max_cars

        self.transition_prob_table = {}
        self.expected_rentals_table = {}
        self.rent_pdf_given_state_transition_table = {}

    def rent_pdf_given_state(self, x, state):
        if x < 0 or x > state:
            return 0

        if x < state:
            return poisson(self.rent_lambda, x)

        # probability that all the cars in the store are rented
        total = 0
        for j in range(state):
            total += poisson(self.rent_lambda, j)
        return 1 - total

    def return_pdf_given_everything(self, x, state, num_rents):
        if x >= 0 and x < self.max_cars - (state - num_rents):
            return poisson(self.return_lambda, x)

        if x < 0 or x > self.max_cars - (state - num_rents):
            return 0

        # probability that so many cars are returned that the lot is full
        total = 0
        for j in range(self.max_cars - (state - num_rents)):
            total += poisson(self.return_lambda, j)
        return 1 - total

    def transition_prob(self, state2, action, state1):
        if (state2, state1 - action) in self.transition_prob_table:
            return self.transition_prob_table[(state2, state1 - action)]

        total = 0
        for k in range( (state1-action) +1):
            total += self.rent_pdf_given_state(k, (state1-action)) * \
                     self.return_pdf_given_everything(state2-(state1-action)+k, state1-action, k)

        self.transition_prob_table[(state2, state1 - action)] = total

        return total

    def expected_rentals_given_state(self, action, state1):
        if state1-action in self.expected_rentals_table:
            return self.expected_rentals_table[state1-action]

        expected_rentals = 0
        for rent in range(self.max_cars):
            expected_rentals += rent * self.rent_pdf_given_state(rent, state1-action)

        self.expected_rentals_table[state1-action] = expected_rentals

        return expected_rentals

    def rent_pdf_given_state_transition(self, x, state2, action, state1):
        if (x, state2, state1-action) in self.rent_pdf_given_state_transition_table:
            return self.rent_pdf_given_state_transition_table[(x, state2, state1-action)]

        likelihood = self.return_pdf_given_everything(state2 - state1 + action + x, state1, x)
        prior = self.rent_pdf_given_state(x, state1 - action)
        normalization = self.transition_prob(state2, action, state1)

        if normalization == 0:
            return 0
        self.rent_pdf_given_state_transition_table[(x, state2, state1-action)] = likelihood * prior / normalization

        return likelihood * prior / normalization

    def expected_rentals_given_state_transition(self, state2, action, state1):
        expected_rentals = 0
        for rent in range(self.max_cars):
            expected_rentals += rent * self.rent_pdf_given_state_transition(rent, state2, action, state1)

        return expected_rentals

class DealershipPolicyIteration:

    def __init__(self, rent_lambda_1, return_lambda_1, rent_lambda_2, return_lambda_2, max_cars, gamma):
        self.lot1 = CarLot(rent_lambda_1, return_lambda_1, max_cars)
        self.lot2 = CarLot(rent_lambda_2, return_lambda_2, max_cars)
        self.gamma = gamma
        self.max_cars = max_cars

    def reward_function_complete_old(self, action, lot1_state1, lot2_state1, lot1_state2, lot2_state2):
        return 10 * (self.lot1.expected_rentals_given_state_transition(lot1_state2, action, lot1_state1) +
                     self.lot2.expected_rentals_given_state_transition(lot2_state2, -action, lot2_state1)) -\
               2 * abs(action)

    def reward_function_complete(self, action, lot1_state1, lot2_state1, lot1_state2, lot2_state2):
        action_cost = abs(action)
        if action > 0: # to account for the worker shuttling one car from lot 1 to lot 2
            action_cost -= 1
        action_cost *= 2

        if lot1_state2 > 10:
            action_cost += 4

        if lot2_state2 > 10:
            action_cost += 4

        return 10 * (self.lot1.expected_rentals_given_state_transition(lot1_state2, action, lot1_state1) +
                     self.lot2.expected_rentals_given_state_transition(lot2_state2, -action, lot2_state1)) - \
               action_cost

    def reward_function(self, action, lot1_state, lot2_state):
        return 10 * (self.lot1.expected_rentals_given_state(action, lot1_state) + self.lot2.expected_rentals_given_state(-action, lot2_state)) - 2 * abs(action)

    def policy_iteration(self, save_models=True):
        policy = np.zeros( (self.max_cars, self.max_cars) ) # initial policy is to just never move any cars around
        oldvalues = np.zeros( (self.max_cars, self.max_cars) )

        keep_iterating = True
        it = 1
        while keep_iterating and it < 100:
            print(f"Now beginning iteration {it}")
            # policy evaluation
            keep_evaluating = True
            while keep_evaluating:
                newvalues = np.zeros( (self.max_cars, self.max_cars) )
                for i in range(self.max_cars):
                    for j in range(self.max_cars):
                        action = round(policy[i, j])
                        reward = self.reward_function(action, i, j)

                        for ni in range(self.max_cars):
                            for nj in range(self.max_cars):
                                newvalues[i, j] += self.lot1.transition_prob(ni, action, i) * \
                                                   self.lot2.transition_prob(nj, -action, j) * \
                                                   (reward + self.gamma * oldvalues[ni, nj])
                delta = np.max(np.abs(newvalues - oldvalues))
                if delta < 0.01:
                    keep_evaluating = False
                oldvalues = newvalues
                print(delta)

            # policy improvement
            newpolicy = np.zeros( (self.max_cars, self.max_cars) )
            for i in range(self.max_cars):
                for j in range(self.max_cars):
                    max_action = -6
                    max_val = -1000000
                    for action in range(-5, 6):

                        totalval = 0
                        for ni in range(self.max_cars):
                            for nj in range(self.max_cars):
                                reward = self.reward_function_complete(action, i, j, ni, nj)
                                totalval += self.lot1.transition_prob(ni, action, i) * \
                                            self.lot2.transition_prob(nj, -action, j) * \
                                            (reward + self.gamma * oldvalues[ni, nj])

                        if totalval > max_val:
                            max_val = totalval
                            max_action = action
                    newpolicy[i, j] = max_action

            print('*'*50)
            print(f"{(policy!=newpolicy).sum()} differences between the old policy and new policy.")
            if (policy == newpolicy).all():
                keep_iterating = False
            print(policy)
            print(newpolicy)
            policy = newpolicy

            it += 1
        if save_models:
            np.save("policy", policy)
            np.save("value", oldvalues)
            print("Value and policy functions saved to disk.")

if __name__ == "__main__":
    rl_problem = DealershipPolicyIteration(rent_lambda_1=3, return_lambda_1=3, rent_lambda_2=4, return_lambda_2=2, max_cars=20, gamma=0.9)
    rl_problem.policy_iteration()

    #
    # reward = rl_problem.reward_function(5, 12, 3)
    #
    # total = 0
    # for ni in range(21):
    #     for nj in range(21):
    #         prob = rl_problem.lot1.transition_prob(ni, 5, 12) * rl_problem.lot2.transition_prob(nj, -5, 3)
    #         total += prob*rl_problem.reward_function_complete(5, 12, 3, ni, nj)
    # print(total)
    # print(reward)
    #
    # lot1 = CarLot(rent_lambda=3, return_lambda=3, max_cars=20)
    # expected_val = lot1.expected_rentals_given_state(3, 5)
    # print(expected_val)
    #
    # state2_probs = [ lot1.transition_prob(x, 3, 5) for x in range(21) ]
    # print(state2_probs)
    # print(sum(state2_probs))
    #
    # transition_expected_vals = [ lot1.expected_rentals_given_state_transition(x, 3, 5) for x in range(21) ]
    # print(transition_expected_vals)
    #
    # total = 0
    # for i in range(21):
    #     total += transition_expected_vals[i] * state2_probs[i]
    # print(total)
    #
    # transition_expected_vals = [ lot1.expected_rentals_given_state_transition(x, 3, 5) for x in range(21) ]
    # print(transition_expected_vals)
    #
    # total = 0
    # for i in range(21):
    #     total += transition_expected_vals[i] * state2_probs[i]
    # print(total)
