from psopt.combination import Combination as optim

# to run this, do not forget to install scikit-learn
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from sklearn.datasets import load_breast_cancer


if __name__ == "__main__":

    # loading breast cancer dataset
    dataset = load_breast_cancer()

    # train-test split
    train_x, test_x, train_Y, test_Y = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=1)

    # create objective function
    def evaluate(solution):
        results = cv(RFC(n_estimators=10), train_x[:, solution], train_Y, cv=3)
        return results["test_score"].mean()

    # instantiate optimizer
    opt = optim(evaluate, list(range(train_x.shape[1])), labels=dataset.feature_names)

    # maximize obj function
    result = opt.maximize(selection_size=15, verbose=True, max_iter=20)

    # result.solution will have the same effect if labels are not provided to the optimizer
    solution = [i for i in range(len(dataset.feature_names)) if dataset.feature_names[i] in result.solution]

    # ======================== COMPARISON ========================

    original = RFC().fit(train_x, train_Y)
    optimized = RFC().fit(train_x[:, solution], train_Y)

    print("\nTest set accuracy\n--------------------------")
    print("All columns:", original.score(test_x, test_Y))
    print("Solution:", optimized.score(test_x[:, solution], test_Y))
