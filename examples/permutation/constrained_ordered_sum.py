from psopt.permutation import PermutationOptimizer as optim

if __name__ == '__main__':
    
    # define objective function: f([a, b, c, ...]) = a/1 + b/2 + c/3 + ...
    def obj_func(x):
        return sum([x[i] / (i+1) for i in range(len(x))])
    
    # list of possible candidates
    candidates = [2,4,11,5,6,3,10,1,7,8,9]
    
    # constraint: sum of values cannot be greater than 16
    constraint = {"fn":sum, "type":">", "value":16}

    # instantiate the optimizer
    opt = optim(obj_func, candidates, constraints=constraint)

    # define a threshold of acceptance for early convergence
    threshold=10
    
    # minimize the obj function
    opt.minimize(selection_size=5, verbose=True, threshold=threshold)
    