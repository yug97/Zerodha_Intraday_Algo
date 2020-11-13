from django.shortcuts import render
import json
import logging
import os
from decimal import Decimal
from datetime import date
from math import floor
import pandas as pd
import ssl
import requests
import threading
import numpy as np
import pandas_ta as ta
from kiteconnect import KiteConnect
import pytz
import datetime
import time
from .variables import HOST, PORT, kite_api_key, kite_api_secret, capital, instrument_dict
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse


intz = pytz.timezone('Asia/Kolkata')
logging.basicConfig(level=logging.DEBUG)
serializer = lambda obj: isinstance(obj, (date, datetime, Decimal)) and str(obj)  # noqa

kite = KiteConnect(api_key=kite_api_key)
# Create a redirect url
redirect_url = "http://{host}:{port}/login".format(host=HOST, port=PORT)

# Login url
login_url = "https://kite.zerodha.com/connect/login?v=3&api_key={api_key}".format(api_key=kite_api_key)

# Kite connect console url
console_url = "https://developers.kite.trade/apps/{api_key}".format(api_key=kite_api_key)

open_orders_stock_names = []

def get_kite_client(request):
    global kite
    kite = KiteConnect(api_key=kite_api_key)
    if "access_token" in request.session:
        kite.set_access_token(request.session["access_token"])
    return kite


def monitor_order(stock_name, stoploss, target, order_id, position_taken,kite):
    curr_order_id = 0
    # buy_time = datetime.datetime.now(tz=intz)
    for item in kite.orders():
        if item.get('parent_order_id') == str(order_id):
            curr_order_id = item.get('order_id')

    if curr_order_id == 0:
        return

    while True:
        ltp = float(kite.ltp(instrument_dict[stock_name])[str(instrument_dict[stock_name])].get('last_price'))
        print(f'LTP : {ltp} \n target : {target} \n stoploss : {stoploss}\n')
        # time_elapsed = datetime.datetime.now(tz=intz) - buy_time
        if position_taken == "long" and ltp >= target:  # or time_elapsed > datetime.timedelta(minutes=90):
            kite.exit_order(variety='co', order_id=curr_order_id, parent_order_id=order_id)
            return "Order Closed"
        elif position_taken == "short" and ltp <= target:  # or time_elapsed > datetime.timedelta(minutes=90):
            kite.exit_order(variety='co', order_id=curr_order_id, parent_order_id=order_id)
            return "Order Closed"
        else:
            print('order for : {} is open'.format(stock_name))
            time.sleep(2)


def index(request):
    data = {
        "api_key":kite_api_key,
        "redirect_url":redirect_url,
        "login_url": login_url
    }
    return render(request, "algo_trading/index.html", {"data":data})


def login(request):
    request_token = request.GET.get("request_token")

    if not request_token:
        return """
            <span style="color: red">
                Error while generating request token.
            </span>
            <a href='/'>Try again.<a>"""

    global kite
    data = kite.generate_session(request_token, api_secret=kite_api_secret)
    request.session["access_token"] = data["access_token"]

    with open("data.json", "r+") as jsonFile:
        data2 = json.load(jsonFile)

    data2["access_token"] = data["access_token"]

    with open("data.json", "w") as jsonFile:
        json.dump(data2, jsonFile)

    context_data = {
        "access_token":data["access_token"],
        "user_data":json.dumps(
            data,
            indent=4,
            sort_keys=True,
            default=serializer
        )
    }
    return render(request, "algo_trading/login.html", {"data":context_data})


@csrf_exempt
def webhook(request):
    global open_orders_stock_names

    with open("data.json", "r") as jsonFile:
        data2 = json.load(jsonFile)

    if len(data2.get("access_token","")):
        request.session["access_token"] = data2.get("access_token")

    kite = KiteConnect(api_key=kite_api_key)
    if "access_token" in request.session:
       kite.set_access_token(request.session["access_token"])

    _capital = capital
    _green_candle = False
    _red_candle = False
    _macd_crossover = False
    _high_volume = False
    _squeeze_momentum = False
    _allowed_position = ["long", "short"]
    last_sq = ""
    lastlast_sq = ""
    stock = ""
    position = ""


    if not json.loads(request.body).get("stock"):
        print(f'jyadaa Load!')
        return HttpResponse("jyadaa Load!")

    stock = json.loads(request.body).get("stock", "").upper()
    position = json.loads(request.body).get("position", "").lower()
    print(f'ALERT: {stock} : {position}')

    if stock not in instrument_dict.keys() \
            or position not in _allowed_position:
        print(f'{stock} galat data bheja ya phir pehle se stock hai open')
        return HttpResponse("galat data bheja ya phir pehle se stock hai open")

    now_ = datetime.datetime.now(tz=intz)
    end_date_for_data = datetime.datetime(now_.year, now_.month, now_.day, 15)
    start_date_for_data = end_date_for_data - datetime.timedelta(days=60)

    df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    try:
        data = kite.historical_data(instrument_dict[stock], start_date_for_data, end_date_for_data,
                                    interval='15minute')
        df = pd.DataFrame.from_dict(data, orient='columns', dtype=None)
        if not df.empty:
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df['date'] = df['date'].astype(str).str[:-6]
            df['date'] = pd.to_datetime(df['date'])
            print(df)
            histdata = df.iloc[-2]
    except Exception as e:
        print("Error in Getting Historical Data", instrument_dict[stock], e)
        return HttpResponse("Error in Fetching Data")

    if histdata['open'] < histdata['close']:
        _green_candle = True
    else:
        _red_candle = True

    if _green_candle:
        if ((histdata['close'] - histdata['open']) > 0.015 * histdata['close']) \
                or ((histdata['high'] - histdata['low']) > 0.019 * histdata['close']):
            print(f'{stock}: Candle is Too Big')
            return HttpResponse("Candle is Too Big")
    elif _red_candle:
        if ((histdata['open'] - histdata['close']) > 0.015 * histdata['close']) \
                or ((histdata['high'] - histdata['low']) > 0.019 * histdata['close']):
            print(f'{stock}: Candle is Too Big')
            return HttpResponse("Candle is Too Big")

    exp1 = df.close.ewm(span=12, adjust=False).mean()
    exp2 = df.close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()
    idx = np.argwhere(np.diff(np.sign(macd - exp3))).flatten()
    list_index = df.index.tolist()
    if idx[-1] in list_index[-11:]:
        _macd_crossover = True
    else:
        print(f'{stock}: MACD crossover Did not happen!')
        return HttpResponse("MACD crossover Did not happen!")

    moving_avarage = df["volume"].tail(50).mean()
    last_vol = df['volume'].iloc[-2]
    lasttolast_vol = df['volume'].iloc[-3]
    if moving_avarage < last_vol and last_vol > lasttolast_vol:
        _high_volume = True
    else:
        print(f'{stock}: Volume is not high!')
        return HttpResponse("Volume is not high!")

    dff = ta.squeeze(df['high'], df['low'], df['close'], bb_length=None, bb_std=None, kc_length=None,
                     kc_scalar=None, mom_length=None, mom_smooth=None, use_tr=None, offset=None, lazybear=True)

    if dff['SQZ_20_2.0_20_1.5_LB'].iloc[-2] > 0:
        if dff['SQZ_20_2.0_20_1.5_LB'].iloc[-2] > dff['SQZ_20_2.0_20_1.5_LB'].iloc[-3]:
            last_sq = "lime"
        else:
            last_sq = "green"
    elif dff['SQZ_20_2.0_20_1.5_LB'].iloc[-2] < 0:
        if dff['SQZ_20_2.0_20_1.5_LB'].iloc[-2] < dff['SQZ_20_2.0_20_1.5_LB'].iloc[-3]:
            last_sq = "red"
        else:
            last_sq = "maroon"

    if dff['SQZ_20_2.0_20_1.5_LB'].iloc[-3] > 0:
        if dff['SQZ_20_2.0_20_1.5_LB'].iloc[-3] > dff['SQZ_20_2.0_20_1.5_LB'].iloc[-4]:
            lastlast_sq = "lime"
        else:
            lastlast_sq = "green"
    elif dff['SQZ_20_2.0_20_1.5_LB'].iloc[-3] < 0:
        if dff['SQZ_20_2.0_20_1.5_LB'].iloc[-3] < dff['SQZ_20_2.0_20_1.5_LB'].iloc[-4]:
            lastlast_sq = "red"
        else:
            lastlast_sq = "maroon"

    if position == "long":
        dictl = {"maroon": 1, "red": 0, "lime": 1, "green": 0}
        if dictl[last_sq] + dictl[lastlast_sq] == 2:
            _squeeze_momentum = True
    elif position == "short":
        dicts = {"maroon": 0, "red": 1, "lime": 0, "green": 1}
        if dicts[last_sq] + dicts[lastlast_sq] == 2:
            _squeeze_momentum = True

    print('Script' + str(stock))
    print('Volume high' + str(_high_volume))
    print('Crossover' + str(_macd_crossover))
    print('Squeeze ' + str(_squeeze_momentum))

    if position == "long" and _green_candle and _high_volume \
            and _macd_crossover and _squeeze_momentum:
        quantity_ = int(floor(max(1, (capital / histdata['close']))))
        stoploss_ = max(histdata['low'], round(histdata['close'] * 0.99, 1))
        buy_target = min((2 * histdata['close'] - histdata['low']), histdata['close'] * 1.03)
        try:
            order_id = kite.place_order(exchange='NSE', tradingsymbol=stock, transaction_type="BUY",
                                        quantity=1,
                                        product='MIS', order_type='MARKET', validity='DAY', trigger_price=stoploss_,
                                        stoploss=stoploss_, variety="co")

            logging.info("Order placed. ID is: {}".format(order_id))
            monitor_order(stock, stoploss_, buy_target, order_id, position,kite)
            #open_orders_stock_names.append(stock)
        except Exception as e:
            logging.info("Order placement failed: {}".format(stock))
            return HttpResponse('Order Processed')

    elif position == 'short' and _red_candle and _high_volume \
            and _macd_crossover and _squeeze_momentum:
        quantity_ = int(floor(max(1, (capital / histdata['close']))))
        stoploss_ = min(histdata['high'], round(histdata['close'] * 1.01, 1))
        sell_target = max((2 * histdata['close'] - histdata['high']), histdata['close'] * 0.97)
        try:
            order_id = kite.place_order(exchange='NSE', tradingsymbol=stock,
                                        transaction_type="SELL", quantity=1,
                                        product='MIS', order_type='MARKET', validity='DAY', trigger_price=stoploss_,
                                        stoploss=stoploss_, variety="co")
            logging.info("Order placed. ID is: {}".format(order_id))
            monitor_order(stock, stoploss_, sell_target, order_id, position,kite)
            #open_orders_stock_names.append(stock)
        except Exception as e:
            logging.info("Order placement failed: {}".format(stock))
            return HttpResponse('Order Processed')
    del df, dff, data
    return HttpResponse("working")
