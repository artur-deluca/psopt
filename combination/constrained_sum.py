from psopt.combination import CombinationOptimizer as optim

if __name__ == '__main__':
    
    # define objective function: f([a, b, c, ...]) = a + b + c + ...
    def obj_func(x): return sum(x)
    
    # list of possible candidates
    candidates = [2,4,11,5,6,3,10,1,7,8,9]
    
    # constraint: sum of values cannot be even
    def mod(x): return sum(x) % 2
    constraint = {"fn":mod, "type":"==", "value":1}

    # instantiate the optimizer
    opt = optim(obj_func, candidates, constraints=constraint)

    # define a threshold of acceptance for early convergence
    threshold=15
    
    # minimize the obj function
    opt.minimize(selection_size=5, verbose=True, threshold=threshold)
    