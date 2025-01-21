class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def should_stop(self):
        return self.early_stop

class SecondEarlyStopping:
    def __init__(self, patience=10, min_delta=0.0002, n_epochs=5):
        
        self.patience = patience
        self.min_delta = min_delta
        self.n_epochs = n_epochs
        self.validation_losses = []
        self.train_losses = []
        self.should_stop = False

    def __call__(self, train_loss, val_loss):
        
        self.train_losses.append(train_loss)
        self.validation_losses.append(val_loss)

        if len(self.validation_losses) >= self.patience:
            if all(val > train for val, train in zip(self.validation_losses[-self.n_epochs:], self.train_losses[-self.n_epochs:])):
                print("Stopping early: Validation loss consistently greater than training loss.")
                self.should_stop = True
                return

            if all(abs(self.validation_losses[i] - self.validation_losses[i + 1]) <= self.min_delta for i in range(len(self.validation_losses) - self.patience, len(self.validation_losses) - 1)):
                print(f"Stopping early: Validation loss decreased by less than {self.min_delta}.")
                self.should_stop = True
                return
    
    def reset(self):

        self.should_stop = False
        self.validation_losses = []
        self.train_losses = []