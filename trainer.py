import random

class Trainer:
    def __init__(self, early_stopping):
        self.early_stopping = early_stopping
        self.loss_history = []

    def train(self, steps=50):
        loss = 1.0

        for step in range(steps):
            # Simulación de entrenamiento
            noise = random.uniform(-0.02, 0.05)
            loss = loss * (0.95 + noise)

            self.loss_history.append(loss)
            print(f"Step {step}: loss={loss:.4f}")

            if self.early_stopping.step(loss):
                print(f"⛔ Early stopping triggered at step {step}")
                break

        return self.loss_history
