import LinRegLearner as lrl, BagLearner as bl

class InsaneLearner(object):

    def __init__(self, verbose = False):
        self.learners = [bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False)]*20
        if verbose: print("\nInitialized Insane Regression:","\nAuthor: ", self.author())

    def author(self): return "jhuang678"

    def add_evidence(self, data_x, data_y):
        for learner in self.learners: learner.add_evidence(data_x, data_y)

    def query(self, points):
        return sum([learner.query(points) for learner in self.learners]) / len(self.learners)

if __name__ == "__main__":
    print("This is InsaneLearner.py. Please use 'InsaneLearner(verbose)' to initialize.")