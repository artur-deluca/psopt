from psopt.combination import CombinationOptimizer as optim

# to run this, do not forget to install scikit-learn
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


if __name__ == '__main__':
    
    # loading breast cancer dataset
    dataset = load_breast_cancer()

    # train-test split
    train_x, test_x, train_Y, test_Y = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=1)

    # train-validation split
    train_x, val_x, train_Y, val_Y = train_test_split(train_x, train_Y, test_size=0.2, random_state=1)

    # create objective function
    def evaluate(solution):
        rfc = RFC().fit(train_x[:, solution], train_Y) 
        return rfc.score(val_x[:, solution], val_Y)

    # instantiate optimizer
    opt = optim(evaluate, list(range(train_x.shape[1])), labels=dataset.feature_names)

    # maximize obj function
    solution = opt.maximize(selection_size=15, verbose=True, max_iter=50, random_state=1)


    # ======================== COMPARISON ========================

    original = RFC().fit(train_x, train_Y)
    optimized = RFC().fit(train_x[:, solution], train_Y)

    print("\nValidation set accuracy\n--------------------------")
    print("All columns:", original.score(val_x, val_Y))
    print("Solution:", optimized.score(val_x[:, solution], val_Y))


    print("\nTest set accuracy\n--------------------------")
    print("All columns:", original.score(test_x, test_Y))
    print("Solution:", optimized.score(test_x[:, solution], test_Y))

