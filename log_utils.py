import logging

def setup_logger(log_dir, logger_name='train_log'):
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    import os
    os.makedirs(log_dir, exist_ok=True)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, ch

def enable_console_logging(logger, console_handler, enable=True):
    """Enable/disable console logging"""
    if enable:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.CRITICAL)