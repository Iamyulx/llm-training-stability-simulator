class EarlyStopping:
    def __init__(self, patience=2, threshold=0.01):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss):
        if loss < self.best_loss - self.threshold:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True # stop training
        return False