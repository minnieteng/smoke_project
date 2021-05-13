import os
import logging
import errno

logging.getLogger(__name__)


def mkdir(dirname):
    """ Check that a directory can be made then make it

    :param dirname: Directory to create
    :type dirname: str or os.path
    :return: True if successful
    :rtype: bool
    """
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            dirname = os.path.dirname(dirname)
            os.makedirs(dirname)
            logging.info(f"utils.utilities.mkdir created directory {dirname}")
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                logging.exception(exc.errno)
                return False
    return True
