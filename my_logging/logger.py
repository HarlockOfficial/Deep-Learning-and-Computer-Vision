import logging


class Logger(object):
    instance: 'Logger' = None

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance') or cls.instance is None:
            cls.instance = super(Logger, cls).__new__(cls)
        return cls.instance

    def __init__(self, min_level=logging.INFO, log_file_path=None):
        logger = logging.getLogger('my_logging')
        logger.setLevel(min_level)
        self.__formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        if log_file_path is not None:
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(min_level)
            fh.setFormatter(self.__formatter)
            logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(min_level)
        ch.setFormatter(self.__formatter)
        logger.addHandler(ch)
        self.__logger = logger

    def debug(self, msg: str):
        self.__logger.debug(msg)

    def info(self, msg: str):
        self.__logger.info(msg)

    def warning(self, msg: str):
        self.__logger.warning(msg)

    def error(self, msg: str):
        self.__logger.error(msg)

    def critical(self, msg: str):
        self.__logger.critical(msg)

    def add_file_path(self, log_level: str, debug_log_file_path: str):
        log_level = self.__parse_log_level(log_level)
        fh = logging.FileHandler(debug_log_file_path)
        fh.setLevel(log_level)
        fh.setFormatter(self.__formatter)
        self.__logger.addHandler(fh)

    def enable(self, log_level: str):
        log_level = self.__parse_log_level(log_level)
        self.__logger.setLevel(log_level)

    @staticmethod
    def __parse_log_level(log_level: str):
        if log_level.lower() == 'debug':
            log_level = logging.DEBUG
        elif log_level.lower() == 'info':
            log_level = logging.INFO
        elif log_level.lower() == 'warning':
            log_level = logging.WARNING
        elif log_level.lower() == 'error':
            log_level = logging.ERROR
        elif log_level.lower() == 'critical':
            log_level = logging.CRITICAL
        else:
            raise ValueError('Invalid log level: ' + log_level)
        return log_level
