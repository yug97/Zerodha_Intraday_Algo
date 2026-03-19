"""Utility functions for the trading algorithm"""
import json
from datetime import datetime, timedelta
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


def serialize_datetime(obj):
    """Serialize datetime and Decimal objects to JSON"""
    if isinstance(obj, (datetime, Decimal)):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def format_json_with_datetime(data):
    """Format data with datetime serialization"""
    return json.dumps(data, indent=4, sort_keys=True, default=serialize_datetime)


def get_market_hours():
    """Get market trading hours"""
    return {
        "market_open": "09:15",
        "market_close": "15:30",
        "pre_market_open": "09:00",
        "pre_market_close": "09:15"
    }


def is_market_open():
    """Check if market is currently open"""
    from datetime import datetime
    import pytz
    
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Market is open Monday to Friday, 9:15 AM to 3:30 PM
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    
    return market_open <= now <= market_close


def calculate_percentage(value, base):
    """Calculate percentage change"""
    if base == 0:
        return 0
    return (value / base) * 100


def format_currency(amount):
    """Format amount as currency"""
    return f"₹{amount:,.2f}"


class ErrorHandler:
    """Centralized error handling"""
    
    @staticmethod
    def log_error(error_type: str, message: str, context: dict = None):
        """Log error with context"""
        if context:
            logger.error(f"{error_type}: {message} | Context: {context}")
        else:
            logger.error(f"{error_type}: {message}")
    
    @staticmethod
    def log_warning(message: str, context: dict = None):
        """Log warning"""
        if context:
            logger.warning(f"{message} | Context: {context}")
        else:
            logger.warning(message)
    
    @staticmethod
    def log_info(message: str, context: dict = None):
        """Log info"""
        if context:
            logger.info(f"{message} | Context: {context}")
        else:
            logger.info(message)
