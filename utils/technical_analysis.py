import numpy as np
import pandas as pd
from scipy import stats

def calculate_ma5_slope(prices):
    """计算5日均线斜率"""
    if len(prices) < 5:
        return None
        
    # 计算5日移动平均线
    ma5 = prices.rolling(window=5).mean()
    
    # 对最近5天的MA5进行线性回归
    x = np.arange(5)
    y = ma5.iloc[-5:].values
    slope, _, _, _, _ = stats.linregress(x, y)
    
    return slope

def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    # 计算价格变化
    delta = prices.diff()
    
    # 分离上涨和下跌
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # 计算RS和RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_volume_change(volumes):
    """计算成交量变化率"""
    if len(volumes) < 5:
        return None
        
    current_volume = volumes.iloc[-1]
    avg_volume = volumes.iloc[-6:-1].mean()
    
    volume_change = (current_volume - avg_volume) / avg_volume
    return volume_change

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    # 计算快线和慢线的EMA
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    # 计算DIF
    dif = ema_fast - ema_slow
    
    # 计算DEA
    dea = dif.ewm(span=signal, adjust=False).mean()
    
    # 计算MACD柱状图
    macd = dif - dea
    
    return pd.DataFrame({
        'DIF': dif,
        'DEA': dea,
        'MACD': macd
    })

def check_buy_signal(stock_data):
    """检查是否满足买入信号"""
    try:
        # 获取收盘价和成交量数据
        prices = stock_data['收盘']
        volumes = stock_data['成交量']
        
        # 1. 计算5日均线斜率
        current_slope = calculate_ma5_slope(prices)
        previous_slope = calculate_ma5_slope(prices.shift(1))
        
        slope_signal = (previous_slope is not None and 
                       current_slope is not None and 
                       previous_slope < 0 and 
                       current_slope > -0.01)  # 放宽斜率条件
        
        if not slope_signal:
            return False, "五日线斜率未转正"
            
        # 2. 检查RSI
        rsi = calculate_rsi(prices)
        current_rsi = rsi.iloc[-1]
        
        if not (25 <= current_rsi <= 75):  # 放宽RSI区间
            return False, "RSI不在合适区间"
            
        # 3. 检查成交量
        volume_change = calculate_volume_change(volumes)
        
        if volume_change is None or volume_change < 0.05:  # 降低成交量放大要求到5%
            return False, "成交量未显著放大"
            
        # 4. 检查MACD
        macd_data = calculate_macd(prices)
        current_dif = macd_data['DIF'].iloc[-1]
        current_dea = macd_data['DEA'].iloc[-1]
        previous_dif = macd_data['DIF'].iloc[-2]
        previous_dea = macd_data['DEA'].iloc[-2]
        
        # 放宽MACD条件，只要DIF上穿DEA即可
        macd_signal = (previous_dif < previous_dea and current_dif > current_dea)
                      
        if not macd_signal:
            return False, "MACD未形成金叉"
            
        # 所有条件都满足
        return True, "满足所有买入条件"
        
    except Exception as e:
        return False, f"计算过程出错: {str(e)}" 