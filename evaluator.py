class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, loss_history):
        if not loss_history:
            return "FAIL"

        if loss_history[-1] > loss_history[0]:
            return "DIVERGED"

        if min(loss_history) == loss_history[-1]:
            return "GOOD"

        return "OK"
