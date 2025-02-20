import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

def calculate_ma5_slope(stock_data):
    """计算MA5斜率"""
    try:
        if stock_data is None or len(stock_data) < 5:
            return 0
            
        # 确保收盘价是数值类型
        prices = pd.to_numeric(stock_data['收盘'], errors='coerce')
        
        # 计算MA5
        ma5 = prices.rolling(window=5).mean()
        
        # 获取最新的5个MA5值
        latest_ma5 = ma5.tail(5).values
        
        # 去除NaN值
        latest_ma5 = latest_ma5[~np.isnan(latest_ma5)]
        
        if len(latest_ma5) < 5:
            return 0
            
        # 使用线性回归计算斜率
        x = np.arange(len(latest_ma5))
        slope, _ = np.polyfit(x, latest_ma5, 1)
        
        return slope
        
    except Exception as e:
        st.error(f"计算MA5斜率时发生错误: {str(e)}")
        return 0

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
        current_slope = calculate_ma5_slope(stock_data)
        previous_slope = calculate_ma5_slope(stock_data.shift(1))
        
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

def plot_stock_chart(stock_code, stock_data):
    """绘制股票K线和MA5"""
    try:
        if stock_data is None or len(stock_data) < 5:
            st.warning("数据不足，无法绘制图表")
            return None
            
        # 确保所有价格列都是数值类型
        numeric_columns = ['开盘', '收盘', '最高', '最低']
        for col in numeric_columns:
            stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
            
        # 计算MA5
        stock_data['MA5'] = stock_data['收盘'].rolling(window=5).mean()
        
        # 创建图表
        fig = go.Figure()
        
        # 添加K线图
        fig.add_trace(go.Candlestick(
            x=stock_data['日期'],
            open=stock_data['开盘'],
            high=stock_data['最高'],
            low=stock_data['最低'],
            close=stock_data['收盘'],
            name='K线'
        ))
        
        # 添加MA5线
        fig.add_trace(go.Scatter(
            x=stock_data['日期'],
            y=stock_data['MA5'],
            name='MA5',
            line=dict(color='orange', width=2)
        ))
        
        # 更新布局
        fig.update_layout(
            title=f'{stock_code} K线图和MA5',
            yaxis_title='价格',
            xaxis_title='日期',
            template='plotly_dark'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"绘制图表时发生错误: {str(e)}")
        return None 