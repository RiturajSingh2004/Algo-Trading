import logging
from datetime import datetime

class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_records = []
    
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'message': self.format(record)
        }
        self.log_records.append(log_entry)
    
    def get_logs(self):
        return self.log_records
    
    def clear_logs(self):
        self.log_records = []

def setup_logger():
    """Configure and return the logger instance."""
    logger = logging.getLogger('algo_trading')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = StreamlitLogHandler()
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger, logger.handlers[0]
