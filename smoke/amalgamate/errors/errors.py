class NoValidValuesInGrid(Exception):
    """ Thrown when given grid is completely filled with nan values

    """

    pass


class IncompletePredictionSet(Exception):
    """ Thrown when any of datasets in prediction set give a no valid value
    grid, meaning set can not be used for prediction.

    """

    pass

class NoCorrespondingLabel(Exception):
    """ Thrown when no label is found for given time

    """

    pass
