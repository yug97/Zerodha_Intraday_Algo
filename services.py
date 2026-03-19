"""Business logic and trading services"""
import json
import logging
import asyncio
import datetime
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytz
from math import floor
from decimal import Decimal
from typing import Dict, Tuple, Optional
from kiteconnect import KiteConnect
from config import settings
from constants import (
    INSTRUMENT_DICT, ALLOWED_POSITIONS, VOLUME_LOOKBACK, MACD_LOOKBACK,
    MIN_CANDLE_SIZE_PCT, MAX_CANDLE_RANGE_PCT, LONG_SL_PCT, SHORT_SL_PCT,
    LONG_TARGET_PCT, SHORT_TARGET_PCT, SQUEEZE_SETTINGS, SQUEEZE_LONG_POSITIVE,
    SQUEEZE_SHORT_POSITIVE
)
from models import AnalysisResult, TradingMetrics

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Timezone
IST = pytz.timezone('Asia/Kolkata')

# Global kite instance
kite_client: Optional[KiteConnect] = None


class KiteService:
    """Service for Kite Connect API interactions"""
    
    def __init__(self):
        self.kite = None
        self.initialize()
    
    def initialize(self):
        """Initialize Kite Connect client"""
        self.kite = KiteConnect(api_key=settings.kite_api_key)
    
    def set_access_token(self, access_token: str):
        """Set access token for authenticated requests"""
        if self.kite:
            self.kite.set_access_token(access_token)
    
    def get_login_url(self) -> str:
        """Get Kite Connect login URL"""
        return f"https://kite.zerodha.com/connect/login?v=3&api_key={settings.kite_api_key}"
    
    def get_redirect_url(self) -> str:
        """Get redirect URL for OAuth callback"""
        return f"http://{settings.redirect_host}:{settings.redirect_port}/login"
    
    def generate_session(self, request_token: str) -> Dict:
        """Generate session from request token"""
        try:
            data = self.kite.generate_session(request_token, api_secret=settings.kite_api_secret)
            return data
        except Exception as e:
            logger.error(f"Session generation failed: {e}")
            raise


class DataService:
    """Service for data management"""
    
    def __init__(self):
        self.data_file = f"{settings.data_dir}/{settings.data_file}"
    
    def load_data(self) -> Dict:
        """Load data from JSON file"""
        try:
            with open(self.data_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"access_token": ""}
    
    def save_data(self, data: Dict):
        """Save data to JSON file"""
        try:
            with open(self.data_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save data: {e}")


class TradingAnalysisService:
    """Service for trading analysis and order placement"""
    
    def __init__(self, kite_service: KiteService, kite: KiteConnect):
        self.kite_service = kite_service
        self.kite = kite
    
    def validate_input(self, stock: str, position: str) -> Tuple[bool, str]:
        """Validate webhook input"""
        if stock.upper() not in INSTRUMENT_DICT:
            return False, f"Invalid stock: {stock}"
        
        if position.lower() not in ALLOWED_POSITIONS:
            return False, f"Invalid position: {position}"
        
        return True, ""
    
    def fetch_historical_data(self, stock: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data"""
        try:
            now = datetime.datetime.now(tz=IST)
            end_date = datetime.datetime(now.year, now.month, now.day, 15)
            start_date = end_date - datetime.timedelta(days=days)
            
            instrument_token = INSTRUMENT_DICT[stock.upper()]
            data = self.kite.historical_data(instrument_token, start_date, end_date, interval='15minute')
            
            df = pd.DataFrame.from_dict(data, orient='columns')
            if df.empty:
                return None
            
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df['date'] = df['date'].astype(str).str[:-6]
            df['date'] = pd.to_datetime(df['date'])
            
            return df
        except Exception as e:
            logger.error(f"Historical data fetch failed for {stock}: {e}")
            return None
    
    def analyze_candle(self, histdata: pd.Series) -> Tuple[bool, bool, str]:
        """Analyze candle characteristics"""
        open_price = histdata['open']
        close_price = histdata['close']
        high_price = histdata['high']
        low_price = histdata['low']
        
        is_green = open_price < close_price
        is_red = open_price > close_price
        
        # Check candle size
        if is_green:
            candle_size = close_price - open_price
            candle_range = high_price - low_price
        else:
            candle_size = open_price - close_price
            candle_range = high_price - low_price
        
        if (candle_size > MIN_CANDLE_SIZE_PCT * close_price or 
            candle_range > MAX_CANDLE_RANGE_PCT * close_price):
            return is_green, is_red, "Candle is too big"
        
        return is_green, is_red, ""
    
    def analyze_macd(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Analyze MACD crossover"""
        try:
            exp1 = df.close.ewm(span=12, adjust=False).mean()
            exp2 = df.close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # Find crossover indices
            crossover_idx = np.argwhere(np.diff(np.sign(macd - signal))).flatten()
            
            if len(crossover_idx) == 0:
                return False, "No MACD crossover found"
            
            last_crossover = crossover_idx[-1]
            list_index = df.index.tolist()
            
            # Check if crossover happened in last 11 candles
            if last_crossover in list_index[-MACD_LOOKBACK:]:
                return True, ""
            
            return False, "MACD crossover did not happen recently"
        except Exception as e:
            logger.error(f"MACD analysis failed: {e}")
            return False, f"MACD analysis error: {str(e)}"
    
    def analyze_volume(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Analyze volume"""
        try:
            moving_avg_volume = df["volume"].tail(VOLUME_LOOKBACK).mean()
            last_volume = df['volume'].iloc[-2]
            prev_volume = df['volume'].iloc[-3]
            
            if moving_avg_volume < last_volume and last_volume > prev_volume:
                return True, ""
            
            return False, "Volume is not high"
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            return False, f"Volume analysis error: {str(e)}"
    
    def analyze_squeeze(self, df: pd.DataFrame) -> Tuple[str, str, str]:
        """Analyze squeeze momentum"""
        try:
            squeeze_df = ta.squeeze(
                df['high'], df['low'], df['close'],
                **SQUEEZE_SETTINGS
            )
            
            # Get squeeze values
            last_squeeze = squeeze_df['SQZ_20_2.0_20_1.5_LB'].iloc[-2]
            prev_squeeze = squeeze_df['SQZ_20_2.0_20_1.5_LB'].iloc[-3]
            prevprev_squeeze = squeeze_df['SQZ_20_2.0_20_1.5_LB'].iloc[-4]
            
            # Determine squeeze colors
            last_sq = self._get_squeeze_color(last_squeeze, prev_squeeze)
            lastlast_sq = self._get_squeeze_color(prev_squeeze, prevprev_squeeze)
            
            return last_sq, lastlast_sq, ""
        except Exception as e:
            logger.error(f"Squeeze analysis failed: {e}")
            return "", "", f"Squeeze analysis error: {str(e)}"
    
    @staticmethod
    def _get_squeeze_color(current: float, previous: float) -> str:
        """Get squeeze momentum color"""
        if current > 0:
            return "lime" if current > previous else "green"
        else:
            return "red" if current < previous else "maroon"
    
    def check_squeeze_momentum(self, last_sq: str, lastlast_sq: str, position: str) -> Tuple[bool, str]:
        """Check squeeze momentum for trading condition"""
        try:
            if position == "long":
                momentum_sum = SQUEEZE_LONG_POSITIVE.get(last_sq, 0) + SQUEEZE_LONG_POSITIVE.get(lastlast_sq, 0)
                if momentum_sum == 2:
                    return True, ""
                return False, "Squeeze momentum not positive for long"
            else:  # short
                momentum_sum = SQUEEZE_SHORT_POSITIVE.get(last_sq, 0) + SQUEEZE_SHORT_POSITIVE.get(lastlast_sq, 0)
                if momentum_sum == 2:
                    return True, ""
                return False, "Squeeze momentum not positive for short"
        except Exception as e:
            logger.error(f"Squeeze momentum check failed: {e}")
            return False, f"Squeeze momentum error: {str(e)}"
    
    def calculate_trading_metrics(self, histdata: pd.Series, position: str) -> Optional[TradingMetrics]:
        """Calculate quantity, stoploss, and target"""
        try:
            entry_price = float(histdata['close'])
            quantity = max(1, int(floor(settings.capital / entry_price)))
            
            if position == "long":
                stoploss = max(
                    float(histdata['low']),
                    round(entry_price * LONG_SL_PCT, 1)
                )
                target = min(
                    2 * entry_price - float(histdata['low']),
                    entry_price * LONG_TARGET_PCT
                )
            else:  # short
                stoploss = min(
                    float(histdata['high']),
                    round(entry_price * SHORT_SL_PCT, 1)
                )
                target = max(
                    2 * entry_price - float(histdata['high']),
                    entry_price * SHORT_TARGET_PCT
                )
            
            return TradingMetrics(
                quantity=quantity,
                stoploss=float(stoploss),
                target=float(target),
                entry_price=entry_price
            )
        except Exception as e:
            logger.error(f"Trading metrics calculation failed: {e}")
            return None
    
    def place_order(self, stock: str, position: str, metrics: TradingMetrics) -> Optional[str]:
        """Place buy or sell order"""
        try:
            transaction_type = "BUY" if position == "long" else "SELL"
            
            order_id = self.kite.place_order(
                exchange='NSE',
                tradingsymbol=stock.upper(),
                transaction_type=transaction_type,
                quantity=metrics.quantity,
                product='MIS',
                order_type='MARKET',
                validity='DAY',
                trigger_price=metrics.stoploss,
                stoploss=metrics.stoploss,
                variety="co"
            )
            
            logger.info(f"Order placed for {stock}: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return None
    
    async def monitor_order(self, stock: str, stoploss: float, target: float, 
                           order_id: str, position: str):
        """Monitor order and close at target or stoploss"""
        try:
            # Find child order ID
            child_order_id = None
            for order in self.kite.orders():
                if order.get('parent_order_id') == str(order_id):
                    child_order_id = order.get('order_id')
                    break
            
            if not child_order_id:
                logger.warning(f"Could not find child order for {stock}")
                return
            
            while True:
                try:
                    instrument_token = INSTRUMENT_DICT[stock.upper()]
                    ltp_data = self.kite.ltp(instrument_token)
                    ltp = float(ltp_data[str(instrument_token)].get('last_price'))
                    
                    logger.info(f"LTP: {ltp}, Target: {target}, Stoploss: {stoploss}")
                    
                    if position == "long" and ltp >= target:
                        self.kite.exit_order(variety='co', order_id=child_order_id, 
                                            parent_order_id=order_id)
                        logger.info(f"Order closed for {stock} at target")
                        break
                    elif position == "short" and ltp <= target:
                        self.kite.exit_order(variety='co', order_id=child_order_id, 
                                            parent_order_id=order_id)
                        logger.info(f"Order closed for {stock} at target")
                        break
                    
                    await asyncio.sleep(2)
                except Exception as e:
                    logger.error(f"Error monitoring order: {e}")
                    await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Order monitoring failed: {e}")
    
    async def process_webhook(self, stock: str, position: str, 
                             access_token: str) -> AnalysisResult:
        """Process webhook and perform analysis"""
        # Validate input
        is_valid, error_msg = self.validate_input(stock, position)
        if not is_valid:
            return AnalysisResult(
                stock=stock, position=position, green_candle=False, red_candle=False,
                macd_crossover=False, high_volume=False, squeeze_momentum=False,
                can_trade=False, reason=error_msg
            )
        
        # Set access token
        self.kite_service.set_access_token(access_token)
        
        # Fetch historical data
        df = self.fetch_historical_data(stock.upper())
        if df is None or len(df) < 4:
            return AnalysisResult(
                stock=stock, position=position, green_candle=False, red_candle=False,
                macd_crossover=False, high_volume=False, squeeze_momentum=False,
                can_trade=False, reason="Failed to fetch historical data"
            )
        
        histdata = df.iloc[-2]
        
        # Analyze candle
        is_green, is_red, candle_error = self.analyze_candle(histdata)
        if candle_error:
            return AnalysisResult(
                stock=stock, position=position, green_candle=is_green, red_candle=is_red,
                macd_crossover=False, high_volume=False, squeeze_momentum=False,
                can_trade=False, reason=candle_error
            )
        
        # Analyze MACD
        macd_crossover, macd_error = self.analyze_macd(df)
        if not macd_crossover:
            return AnalysisResult(
                stock=stock, position=position, green_candle=is_green, red_candle=is_red,
                macd_crossover=macd_crossover, high_volume=False, squeeze_momentum=False,
                can_trade=False, reason=macd_error
            )
        
        # Analyze volume
        high_volume, volume_error = self.analyze_volume(df)
        if not high_volume:
            return AnalysisResult(
                stock=stock, position=position, green_candle=is_green, red_candle=is_red,
                macd_crossover=macd_crossover, high_volume=high_volume, squeeze_momentum=False,
                can_trade=False, reason=volume_error
            )
        
        # Analyze squeeze
        last_sq, lastlast_sq, squeeze_error = self.analyze_squeeze(df)
        if squeeze_error:
            return AnalysisResult(
                stock=stock, position=position, green_candle=is_green, red_candle=is_red,
                macd_crossover=macd_crossover, high_volume=high_volume, squeeze_momentum=False,
                can_trade=False, reason=squeeze_error
            )
        
        # Check squeeze momentum
        squeeze_momentum, squeeze_momentum_error = self.check_squeeze_momentum(last_sq, lastlast_sq, position)
        
        # Determine if we can trade
        can_trade = False
        reason = None
        
        if position == "long" and is_green and macd_crossover and high_volume and squeeze_momentum:
            can_trade = True
        elif position == "short" and is_red and macd_crossover and high_volume and squeeze_momentum:
            can_trade = True
        else:
            reason = "Not all trading conditions met"
        
        return AnalysisResult(
            stock=stock, position=position, green_candle=is_green, red_candle=is_red,
            macd_crossover=macd_crossover, high_volume=high_volume, 
            squeeze_momentum=squeeze_momentum, can_trade=can_trade, reason=reason
        )


# Initialize services
kite_service = KiteService()
data_service = DataService()
