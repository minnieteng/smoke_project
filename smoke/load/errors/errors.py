class DimensionError(Exception):
    """ Thrown when there is an error in the dimensions of the dataset given,
    i.e. coords are not time, lat, lon only.

    """

    pass


class WrongTypeError(Exception):
    """ Thrown when wrong data type for data set is given.

    """

    pass


class FeatureNotInDataset(Exception):
    """ Thrown if given feature name not in dataset.

    """

    pass
