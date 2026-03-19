"""FastAPI application for Zerodha Intraday Trading Algorithm"""
import json
import logging
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from kiteconnect import KiteConnect

from config import settings
from models import (
    WebhookRequest, LoginRequest, IndexResponse, LoginResponse,
    WebhookResponse, OrderPlacementResponse
)
from services import (
    kite_service, data_service, TradingAnalysisService
)
from constants import INSTRUMENT_DICT

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Zerodha Intraday Trading Algorithm",
    description="Automated trading algorithm using technical indicators",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trading service
trading_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global trading_service
    kite_instance = KiteConnect(api_key=settings.kite_api_key)
    trading_service = TradingAnalysisService(kite_service, kite_instance)
    logger.info("Trading service initialized")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Index endpoint - returns HTML with login information"""
    try:
        login_url = kite_service.get_login_url()
        redirect_url = kite_service.get_redirect_url()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Zerodha Intraday Trading Algorithm</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                    text-align: center;
                    max-width: 500px;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 10px;
                }}
                p {{
                    color: #666;
                    margin-bottom: 30px;
                }}
                a {{
                    display: inline-block;
                    padding: 12px 30px;
                    background-color: #667eea;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: bold;
                    transition: background-color 0.3s;
                }}
                a:hover {{
                    background-color: #764ba2;
                }}
                .info {{
                    background: #f0f0f0;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 20px;
                    text-align: left;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Zerodha Intraday Trading Algorithm</h1>
                <p>Automated trading with technical indicators</p>
                <a href="{login_url}">Login with Kite Connect</a>
                <div class="info">
                    <strong>API Key:</strong> {settings.kite_api_key}<br>
                    <strong>Redirect URL:</strong> {redirect_url}<br>
                </div>
            </div>
        </body>
        </html>
        """
        return html_content
    except Exception as e:
        logger.error(f"Index endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/index", response_model=IndexResponse)
async def api_index():
    """API endpoint for index data"""
    try:
        return IndexResponse(
            api_key=settings.kite_api_key,
            redirect_url=kite_service.get_redirect_url(),
            login_url=kite_service.get_login_url()
        )
    except Exception as e:
        logger.error(f"API index error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/login", response_class=HTMLResponse)
async def login(request_token: str = None):
    """Login endpoint - handles OAuth callback"""
    try:
        if not request_token:
            error_html = """
            <html>
            <body style="display: flex; justify-content: center; align-items: center; height: 100vh;">
                <div style="text-align: center;">
                    <h1 style="color: red;">Error</h1>
                    <p>No request token provided</p>
                    <a href="/">Try again</a>
                </div>
            </body>
            </html>
            """
            return error_html
        
        # Generate session
        session_data = kite_service.generate_session(request_token)
        access_token = session_data.get("access_token")
        
        # Save to data file
        data = data_service.load_data()
        data["access_token"] = access_token
        data_service.save_data(data)
        
        # Set access token in kite service
        kite_service.set_access_token(access_token)
        
        # Format user data
        user_data_str = json.dumps(session_data, indent=2, default=str)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Login Success</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                    max-width: 600px;
                }}
                h1 {{
                    color: #28a745;
                    margin-bottom: 20px;
                }}
                .token {{
                    background: #f0f0f0;
                    padding: 15px;
                    border-radius: 5px;
                    word-break: break-all;
                    margin-bottom: 20px;
                }}
                .label {{
                    font-weight: bold;
                    color: #333;
                    margin-top: 10px;
                }}
                pre {{
                    background: #f0f0f0;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>✅ Login Successful!</h1>
                <div class="label">Access Token:</div>
                <div class="token">{access_token}</div>
                <div class="label">User Data:</div>
                <pre>{user_data_str}</pre>
                <a href="/" style="display: inline-block; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 5px;">Back to Home</a>
            </div>
        </body>
        </html>
        """
        return html_content
    except Exception as e:
        logger.error(f"Login error: {e}")
        error_html = f"""
        <html>
        <body style="display: flex; justify-content: center; align-items: center; height: 100vh;">
            <div style="text-align: center;">
                <h1 style="color: red;">Login Failed</h1>
                <p>{str(e)}</p>
                <a href="/">Try again</a>
            </div>
        </body>
        </html>
        """
        return error_html


@app.post("/webhook", response_model=WebhookResponse)
async def webhook(request: WebhookRequest, background_tasks: BackgroundTasks):
    """
    Webhook endpoint for trading alerts
    
    Receives stock symbol and position (long/short), analyzes technical indicators,
    and places orders if all conditions are met.
    """
    try:
        logger.info(f"Webhook received: {request.stock} - {request.position}")
        
        # Load access token from file
        data = data_service.load_data()
        access_token = data.get("access_token", "")
        
        if not access_token:
            raise HTTPException(status_code=401, detail="Not authenticated. Please login first.")
        
        # Perform analysis
        analysis_result = await trading_service.process_webhook(
            request.stock, request.position, access_token
        )
        
        # If we can trade, place order
        order_id = None
        if analysis_result.can_trade:
            df = trading_service.fetch_historical_data(request.stock.upper())
            if df is not None and len(df) >= 2:
                histdata = df.iloc[-2]
                metrics = trading_service.calculate_trading_metrics(histdata, request.position)
                
                if metrics:
                    order_id = trading_service.place_order(request.stock.upper(), request.position, metrics)
                    
                    # Start monitoring in background
                    if order_id:
                        background_tasks.add_task(
                            trading_service.monitor_order,
                            request.stock.upper(),
                            metrics.stoploss,
                            metrics.target,
                            order_id,
                            request.position
                        )
        
        return WebhookResponse(
            status="success" if analysis_result.can_trade else "analysis_complete",
            message=analysis_result.reason or "Analysis completed",
            stock=request.stock,
            position=request.position,
            order_id=order_id
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")


@app.get("/api/stocks")
async def get_stocks():
    """Get list of supported stocks"""
    try:
        return {
            "count": len(INSTRUMENT_DICT),
            "stocks": list(INSTRUMENT_DICT.keys())
        }
    except Exception as e:
        logger.error(f"Error fetching stocks: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Zerodha Intraday Trading Algorithm",
        "version": "1.0.0"
    }


@app.get("/api/config")
async def get_config():
    """Get trading configuration"""
    return {
        "capital": settings.capital,
        "api_key": settings.kite_api_key,
        "host": settings.host,
        "port": settings.port,
        "redirect_url": kite_service.get_redirect_url()
    }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def server_error_handler(request, exc):
    """Custom 500 handler"""
    logger.error(f"Server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower()
    )
