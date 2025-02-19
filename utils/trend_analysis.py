import numpy as np
import pandas as pd

def calculate_trend_indicators(df):
    """计算趋势指标"""
    try:
        # 计算多个周期的移动平均线
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA10'] = df['收盘'].rolling(window=10).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['MA60'] = df['收盘'].rolling(window=60).mean()
        
        # 计算趋势强度指标(ADX)
        df['TR'] = np.maximum(
            df['最高'] - df['最低'],
            np.maximum(
                abs(df['最高'] - df['收盘'].shift(1)),
                abs(df['最低'] - df['收盘'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # 计算动量指标
        df['ROC'] = (df['收盘'] - df['收盘'].shift(10)) / df['收盘'].shift(10) * 100
        
        # 计算布林带
        df['BB_middle'] = df['收盘'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['收盘'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['收盘'].rolling(window=20).std()
        
        return df
        
    except Exception as e:
        print(f"计算趋势指标时发生错误: {str(e)}")
        return None

def check_trend_signal(stock_data):
    """检查趋势信号"""
    try:
        df = calculate_trend_indicators(stock_data)
        if df is None:
            return False, "计算指标失败"
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        score = 0
        
        # 1. 均线多头排列（权重：30%）
        if (latest['MA5'] > latest['MA10'] > latest['MA20'] > latest['MA60']):
            signals.append("均线多头排列")
            score += 30
        
        # 2. 价格位置（权重：20%）
        if latest['收盘'] > latest['MA5']:
            signals.append("价格站上5日线")
            score += 10
        if latest['收盘'] > latest['BB_upper']:
            signals.append("价格突破布林上轨")
            score += 10
            
        # 3. 趋势强度（权重：20%）
        price_trend = (df['收盘'].iloc[-5:] > df['收盘'].iloc[-5:].shift(1)).sum()
        if price_trend >= 3:
            signals.append("近期价格走势强势")
            score += 20
            
        # 4. 成交量（权重：15%）
        vol_ma5 = df['成交量'].rolling(window=5).mean().iloc[-1]
        if latest['成交量'] > vol_ma5 * 1.2:
            signals.append("成交量放大")
            score += 15
            
        # 5. 动量（权重：15%）
        if latest['ROC'] > 0:
            signals.append("动量指标为正")
            score += 15
            
        # 综合评分和信号
        if score >= 70:  # 设置70分为买入阈值
            return True, f"趋势信号强度: {score}分 - " + ", ".join(signals)
        else:
            return False, f"信号强度不足: {score}分"
            
    except Exception as e:
        return False, f"信号检查失败: {str(e)}"

def get_trend_description():
    """获取策略说明"""
    return """
    ### 传统趋势跟踪策略
    
    #### 策略概述
    本策略基于技术分析中的趋势跟踪理论，通过多个技术指标的综合分析来识别强势股票。
    
    #### 核心指标
    1. 均线系统（30分）
       - 考察MA5、MA10、MA20、MA60的排列
       - 判断多头排列形态
    
    2. 价格位置（20分）
       - 相对于均线位置
       - 布林带突破情况
    
    3. 趋势强度（20分）
       - 考察近期价格连续上涨情况
       - 结合ATR衡量波动
    
    4. 成交量（15分）
       - 对比5日平均成交量
       - 关注量能配合
    
    5. 动量指标（15分）
       - ROC指标
       - 衡量上涨动能
    
    #### 买入条件
    - 综合得分达到70分以上
    - 各项指标配合良好
    - 符合趋势跟踪特征
    
    #### 风险控制
    - 排除ST股票
    - 考虑流动性因素
    - 设置止损位置
    """ 