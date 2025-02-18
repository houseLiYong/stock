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

def ma5_strategy():
    """五日线斜率策略页面"""
    st.title("五日线斜率转正策略")
    
    # 添加策略说明
    with st.expander("策略说明"):
        st.write("""
        ### 策略概述
        本策略通过五日线斜率由负转正的信号，结合其他技术指标，筛选具有潜在上涨动能的股票。
        
        ### 选股条件
        1. 排除ST股票
        2. 五日线斜率由负转正
        3. RSI在30-70之间
        4. 成交量显著放大
        5. MACD金叉且柱状图由负转正
        """)

    # 创建选股按钮
    if st.button("开始选股"):
        with st.spinner("正在进行选股分析..."):
            result_df = get_stock_list()
            
            if result_df is not None and not result_df.empty:
                # 显示选股结果
                st.write("### 选股结果")
                st.dataframe(
                    result_df,
                    column_config={
                        "代码": st.column_config.TextColumn("代码"),
                        "名称": st.column_config.TextColumn("名称"),
                        "现价": st.column_config.NumberColumn("现价", format="%.2f"),
                        "涨跌幅": st.column_config.NumberColumn("涨跌幅", format="%.2f%%"),
                        "换手率": st.column_config.NumberColumn("换手率", format="%.2f%%"),
                        "成交额": st.column_config.NumberColumn("成交额", format="%.2f亿"),
                        "选股理由": st.column_config.TextColumn("选股理由"),
                    },
                    hide_index=True
                )
                
                st.dataframe(result_df)
                # 创建股票选择器
                selected_stock = st.selectbox(
                    "选择要查看的股票",
                    options=result_df['代码'].tolist(),
                    format_func=lambda x: f"{x} - {result_df[result_df['代码']==x]['名称'].iloc[0]}"
                )
                # 确保选中的代码仍然存在于 result_df
                if selected_stock in result_df['代码'].values:
                    stock_name = result_df.loc[result_df['代码'] == selected_stock, '名称'].values[0]
                    st.write(f"你选择的股票是：{selected_stock} - {stock_name}")
                else:
                    st.warning("选中的股票已不存在，请重新选择！")
                print("股票代码",selected_stock)
                if selected_stock:
                    # 获取股票数据
                    stock_data = get_stock_data(selected_stock)
                    
                    if stock_data is not None:
                        # 显示K线图
                        st.plotly_chart(plot_stock_chart(selected_stock, stock_data))
                        
                        # 显示技术指标
                        st.plotly_chart(plot_technical_indicators(stock_data))
                        
                        # 显示详细分析
                        st.write("### 技术指标分析")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("MA5斜率分析")
                            slope = calculate_ma5_slope(stock_data['收盘'])
                            st.write(f"当前斜率: {slope:.4f}")
                            
                            st.write("RSI分析")
                            rsi = calculate_rsi(stock_data['收盘']).iloc[-1]
                            st.write(f"当前RSI: {rsi:.2f}")
                        
                        with col2:
                            st.write("成交量分析")
                            volume_change = calculate_volume_change(stock_data['成交量'])
                            st.write(f"成交量变化率: {volume_change:.2%}")
                            
                            st.write("MACD分析")
                            macd_data = calculate_macd(stock_data['收盘'])
                            st.write(f"DIF: {macd_data['DIF'].iloc[-1]:.4f}")
                            st.write(f"DEA: {macd_data['DEA'].iloc[-1]:.4f}")
                            st.write(f"MACD: {macd_data['MACD'].iloc[-1]:.4f}")
            else:
                st.warning("未找到符合条件的股票")
    
    # 显示策略统计信息
    if 'result_df' in locals() and result_df is not None and not result_df.empty:
        try:
            st.write(f"""
            ### 策略统计
            - 符合条件股票数量: {len(result_df)}
            - 平均涨跌幅: {result_df['涨跌幅'].mean():.2f}%
            - 平均换手率: {result_df['换手率'].mean():.2f}%
            """)
        except Exception as e:
            st.error(f"计算统计信息时发生错误: {str(e)}")
            st.write(f"""
            ### 策略统计
            - 符合条件股票数量: {len(result_df)}
            """)

if __name__ == "__main__":
    ma5_strategy() 