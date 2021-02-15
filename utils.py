import logging


def dict_to_str(d):
    return ", ".join([f"{k}: {round(v, 2)}" for k, v in d.items()])


def get_logger(logger, log_file_path):
    logger.setLevel(logging.DEBUG)
    msg_fmt = "%(asctime)s - %(levelname)-5s - %(name)s -   %(message)s"
    date_fmt = "%m/%d/%Y %H:%M:%S"

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt=msg_fmt, datefmt=date_fmt)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


