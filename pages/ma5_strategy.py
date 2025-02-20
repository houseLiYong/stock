import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from stock.utils.stock_data import get_stock_list, get_stock_data
from stock.utils.technical_analysis import (
    calculate_ma5_slope,
    calculate_rsi,
    calculate_macd,
    calculate_volume_change
)
from plotly.subplots import make_subplots

def plot_stock_chart(stock_code, stock_data):
    """绘制股票图表"""
    try:
         # 确保日期列是正确的datetime类型
        stock_data = stock_data.copy()  # 创建副本避免修改原始数据
        stock_data['日期'] = pd.to_datetime(stock_data['日期'], format='%Y-%m-%d')
        stock_data.set_index('日期', inplace=True)
        
    
        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,  # 2行1列的布局
            shared_xaxes=True,  # 共享X轴
            vertical_spacing=0.05,  # 垂直间距
            row_heights=[0.7, 0.3],  # K线图占70%，成交量占30%
            subplot_titles=(f'{stock_code} K线图', '成交量')
        )
        
        # 添加K线图
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['开盘'],
                high=stock_data['最高'],
                low=stock_data['最低'],
                close=stock_data['收盘'],
                name='K线'
            ),
            row=1, col=1
        )
        
        # 添加MA5
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['收盘'].rolling(window=5).mean(),
                name='MA5',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # 添加成交量图
        colors = ['red' if row['收盘'] >= row['开盘'] else 'green' 
                 for _, row in stock_data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['成交量'],
                name='成交量',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title=f'{stock_code} 行情图',
            height=800,  # 增加总高度
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False  # 禁用rangeslider
        )
        
        # 更新X轴
        fig.update_xaxes(
            title='日期',
            type='date',
            tickformat='%Y-%m-%d',
            tickangle=-45,
            tickmode='auto',
            nticks=10,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        )
        
        # 更新K线图Y轴
        fig.update_yaxes(
            title='价格',
            tickformat='.2f',
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        )
        
        # 更新成交量Y轴
        fig.update_yaxes(
            title='成交量',
            tickformat='.0f',
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        )
        
        # 添加交互功能
        fig.update_layout(
            hovermode='x unified',
            hoverdistance=100,
            spikedistance=1000,
        )
        
        fig.update_xaxes(
            showspikes=True,
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1,
        )
        
        fig.update_yaxes(
            showspikes=True,
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1,
        )
        return fig
        
    except Exception as e:
        return f"绘制图表时发生错误: {str(e)}"

def plot_technical_indicators(stock_data):
    """绘制技术指标图"""
    try:
        # 计算技术指标
        rsi = calculate_rsi(stock_data['收盘'])
        macd_data = calculate_macd(stock_data['收盘'])
        
        # 创建RSI图
        rsi_line = go.Scatter(
            x=stock_data.index,
            y=rsi,
            name='RSI'
        )
        
        # 创建MACD图
        macd_line = go.Scatter(
            x=stock_data.index,
            y=macd_data['MACD'],
            name='MACD'
        )
        dif_line = go.Scatter(
            x=stock_data.index,
            y=macd_data['DIF'],
            name='DIF'
        )
        dea_line = go.Scatter(
            x=stock_data.index,
            y=macd_data['DEA'],
            name='DEA'
        )
        
        # 创建子图
        fig = go.Figure()
        fig.add_trace(rsi_line)
        fig.add_trace(macd_line)
        fig.add_trace(dif_line)
        fig.add_trace(dea_line)
        
        # 设置布局
        fig.update_layout(
            title='技术指标',
            height=400,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        return f"绘制技术指标时发生错误: {str(e)}"

def select_boards():
    """处理板块选择逻辑"""
    with st.sidebar:
        st.header("板块选择")
        st.write("请选择要分析的板块：")
        
        # 预定义的板块列表
        all_boards = [
            '上证主板',
            '深证主板',
            '创业板',
            '科创板',
            '北交所'
        ]
        
        # 使用 checkbox 代替 multiselect，更适合侧边栏
        selected_boards = []
        for board in all_boards:
            if st.checkbox(board, value=True if board in ['上证主板', '创业板', '科创板'] else False, key=f"ma5_{board}"):
                selected_boards.append(board)
        
        # 显示已选板块数量
        if selected_boards:
            st.success(f"已选择 {len(selected_boards)} 个板块")
        else:
            st.warning("请至少选择一个板块")
        
        # 添加刷新按钮
        if st.button("刷新数据", key="ma5_refresh_data"):
            st.cache_data.clear()
            st.rerun()  # 使用 st.rerun() 替代 st.experimental_rerun()
            
        return selected_boards

def ma5_strategy():
    """MA5策略页面"""
    # 首先获取选择的板块
    selected_boards = select_boards()
    
    # 然后显示主页面内容
    st.title("MA5均线策略")
    
    # 显示策略说明
    with st.expander("策略说明"):
        st.markdown("""
        ### MA5均线策略
        
        #### 策略概述
        本策略基于5日均线（MA5）的变化趋势进行选股，主要关注均线的斜率变化和价格位置。
        
        #### 核心指标
        1. MA5斜率
           - 计算最近5日MA5的斜率
           - 判断趋势方向和强度
        
        2. 价格位置
           - 相对MA5的位置
           - 突破确认
        
        3. 成交量
           - 对比5日平均成交量
           - 关注量能配合
        
        #### 买入条件
        - MA5斜率为正且增大
        - 价格站上MA5
        - 成交量放大
        
        #### 风险控制
        - 排除ST股票
        - 考虑流动性因素
        - 设置止损位置
        """)
    
    # 创建选股按钮
    if st.button("开始选股", key="ma5_start_analysis"):
        if not selected_boards:
            st.warning("请先在左侧边栏选择至少一个板块")
            return
            
        with st.spinner("正在进行MA5策略分析..."):
            result_df = get_stock_list(selected_boards)
            
            if result_df is not None and not result_df.empty:
                # 显示选股结果
                st.write("### 选股结果")
                st.dataframe(
                    result_df,
                    column_config={
                        "代码": st.column_config.TextColumn("代码"),
                        "名称": st.column_config.TextColumn("名称"),
                        "板块": st.column_config.TextColumn("板块"),
                        "现价": st.column_config.NumberColumn("现价", format="%.2f"),
                        "涨跌幅": st.column_config.NumberColumn("涨跌幅", format="%.2f%%"),
                        "换手率": st.column_config.NumberColumn("换手率", format="%.2f%%"),
                        "成交额": st.column_config.NumberColumn("成交额", format="%.2f亿"),
                        "选股理由": st.column_config.TextColumn("选股理由"),
                    }
                )
                
                # 创建股票选择器
                selected_stock = st.selectbox(
                    "选择要查看的股票",
                    options=result_df['代码'].tolist(),
                    format_func=lambda x: f"{x} - {result_df[result_df['代码']==x]['名称'].iloc[0]}",
                    key="ma5_stock_selector"
                )
                
                if selected_stock:
                    stock_data = get_stock_data(selected_stock)
                    if stock_data is not None:
                        # 显示MA5分析图表
                        fig = plot_stock_chart(selected_stock, stock_data)
                        if fig:
                            st.plotly_chart(fig)
                            
                        # 显示MA5斜率分析
                        slope = calculate_ma5_slope(stock_data)
                        st.write("### MA5斜率分析")
                        st.write(f"当前MA5斜率: {slope:.4f}")
                        
                        if slope > 0:
                            st.success("MA5斜率为正，趋势向上")
                        else:
                            st.warning("MA5斜率为负，趋势向下")

if __name__ == "__main__":
    ma5_strategy() 