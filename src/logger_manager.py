import logging
import os


class LoggerHandler:
    def __init__(self,path):
        self.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        self.path = path
        self.default_logger = None
        self.loggers = {}

    def create_logger(self, logger_name, file_name):
        if logger_name not in self.loggers.keys():
            handler = logging.FileHandler(os.path.join(self.path, file_name))
            handler.setFormatter(self.formatter)
            logger = logging.getLogger(logger_name)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            self.loggers[logger_name] = logger
        return self.loggers[logger_name]

    def get_logger(self, logger_name):
        return self.loggers[logger_name]

    def set_default_logger(self, logger_name):
        self.default_logger = self.get_logger(logger_name)

    def log(self, message):
        self.default_logger.info(message)
        print(message)

