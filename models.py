"""Pydantic models for request and response validation"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class WebhookRequest(BaseModel):
    """Webhook request model"""
    stock: str = Field(..., description="Stock symbol (e.g., 'INFY')")
    position: str = Field(..., description="Position type: 'long' or 'short'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "stock": "INFY",
                "position": "long"
            }
        }


class LoginRequest(BaseModel):
    """Login request model"""
    request_token: str = Field(..., description="Request token from Kite Connect")


class IndexResponse(BaseModel):
    """Index page response"""
    api_key: str
    redirect_url: str
    login_url: str


class LoginResponse(BaseModel):
    """Login response"""
    access_token: str
    user_data: Dict[str, Any]


class OrderPlacementResponse(BaseModel):
    """Order placement response"""
    status: str
    message: str
    order_id: Optional[str] = None


class WebhookResponse(BaseModel):
    """Webhook response"""
    status: str
    message: str
    stock: Optional[str] = None
    position: Optional[str] = None
    order_id: Optional[str] = None


class AnalysisResult(BaseModel):
    """Trade analysis result"""
    stock: str
    position: str
    green_candle: bool
    red_candle: bool
    macd_crossover: bool
    high_volume: bool
    squeeze_momentum: bool
    can_trade: bool
    reason: Optional[str] = None


class TradingMetrics(BaseModel):
    """Trading metrics"""
    quantity: int
    stoploss: float
    target: float
    entry_price: float
