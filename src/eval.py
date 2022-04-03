

class BaseEvaluator:
    """
    This is the base evaluator class that implements some general methods that every evaluator shall have
    """

    def __init__(self):
        """
        Constructor method of the Base evaluator
        """
        a = 0

    def evaluate(self, model, eval_set):
        """
        This method is abstract and has to be implemented by

        :param model: the model to evaluate
        :param eval_set: the data set for evaluation
        :return: the score of evaluation
        """
        pass


class DiceEvaluator(BaseEvaluator):
    """
    This evaluator is based on the dice score

    TODO: Some notes from the paper that we shall output
        - Dice Score
        - Confidence Interval
        - k-fold cross validation
    """

    def evaluate(self, model, eval_set):
        """
        Evaluate the model based on the dice score
        :return:
        """

        # TODO: do some dice score computations

        # Return the dice score
        return 100
