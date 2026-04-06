from early_stopping import EarlyStopping
from trainer import Trainer
from evaluator import Evaluator

def run_simulation():
    early_stopping = EarlyStopping(patience=2, threshold=0.01)
    trainer = Trainer(early_stopping)
    evaluator = Evaluator()

    loss_history = trainer.train(steps=50)
    result = evaluator.evaluate(loss_history)

    print("\n📊 Evaluation Result:", result)

if __name__ == "__main__":
    run_simulation()