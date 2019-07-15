import logging
import os

def make_logger(name, verbose):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    if verbose == 1:
        # only set the console handler level
        consoleHandler.setLevel(logging.INFO)
    
    elif verbose == 2:
        # set the console handler level
        consoleHandler.setLevel(logging.INFO)

        # make sure that fileHandler logging directory exists
        os.makedirs(os.path.join(os.getcwd(), '.logs'), exist_ok=True)

        # add fileHandler in logger
        fileHandler = logging.FileHandler('.logs/logging.log')
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fileHandler)
    
    else:
        consoleHandler.setLevel(logging.WARNING)
    
    logger.addHandler(consoleHandler)

    return logger