"""Configuration module for Zerodha Intraday Algo"""
from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
    # Kite Connect API Credentials
    kite_api_key: str = "t5dpjo9d4s3l6ax7"
    kite_api_secret: str = "n1v5fdlg4qt9a9hhtuh970j1wuusf156"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Redirect Configuration
    redirect_host: str = "localhost"
    redirect_port: int = 8000
    
    # Trading Configuration
    capital: float = 200000.0
    
    # Data Configuration
    data_dir: str = str(Path(__file__).parent)
    data_file: str = "data.json"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
