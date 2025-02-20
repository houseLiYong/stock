import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from stock.utils.stock_data import get_stock_data, get_stock_list
from stock.utils.trend_analysis import (
    calculate_trend_indicators,
    check_trend_signal,
    get_trend_description
)

def plot_trend_chart(stock_code, stock_data):
    """绘制趋势分析图表"""
    try:
        df = calculate_trend_indicators(stock_data)
        if df is None:
            return None
            
        # 创建子图
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{stock_code} 趋势分析', 'ROC动量', '成交量')
        )
        
        # 1. K线图和均线
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['开盘'],
                high=df['最高'],
                low=df['最低'],
                close=df['收盘'],
                name='K线'
            ),
            row=1, col=1
        )
        
        # 添加均线
        colors = ['blue', 'orange', 'purple', 'gray']
        mas = ['MA5', 'MA10', 'MA20', 'MA60']
        
        for ma, color in zip(mas, colors):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ma],
                    name=ma,
                    line=dict(color=color, width=1)
                ),
                row=1, col=1
            )
            
        # 添加布林带
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_upper'],
                name='布林上轨',
                line=dict(color='gray', dash='dash', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_lower'],
                name='布林下轨',
                line=dict(color='gray', dash='dash', width=1),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # 2. ROC
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ROC'],
                name='ROC',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        # 3. 成交量
        colors = ['red' if row['收盘'] >= row['开盘'] else 'green' 
                 for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['成交量'],
                name='成交量',
                marker_color=colors
            ),
            row=3, col=1
        )
        
        # 更新布局
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False
        )
        
        # 更新坐标轴
        fig.update_xaxes(title_text="日期", row=3, col=1)
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="ROC", row=2, col=1)
        fig.update_yaxes(title_text="成交量", row=3, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"绘制图表时发生错误: {str(e)}")
        return None

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
            if st.checkbox(board, value=True if board in ['上证主板', '创业板', '科创板'] else False, key=f"trend_{board}"):
                selected_boards.append(board)
        
        # 显示已选板块数量
        if selected_boards:
            st.success(f"已选择 {len(selected_boards)} 个板块")
        else:
            st.warning("请至少选择一个板块")
        
        # 添加刷新按钮
        if st.button("刷新数据", key="trend_refresh_data"):
            st.cache_data.clear()
            st.rerun()  # 使用 st.rerun() 替代 st.experimental_rerun()
            
        return selected_boards

def trend_strategy():
    """趋势跟踪策略页面"""
    st.title("趋势跟踪策略")
    
    # 显示策略说明
    with st.expander("策略说明"):
        st.markdown(get_trend_description())
    
    # 获取选择的板块
    selected_boards = select_boards()
    
    # 创建选股按钮
    if selected_boards and st.button("开始选股", key="trend_start_analysis"):
        with st.spinner("正在进行趋势分析..."):
            result_df = get_stock_list(selected_boards)  # 传入选择的板块
            
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
                    key="trend_stock_selector"
                )
                
                if selected_stock:
                    stock_data = get_stock_data(selected_stock)
                    if stock_data is not None:
                        # 显示趋势分析图表
                        st.plotly_chart(plot_trend_chart(selected_stock, stock_data))
                        
                        # 显示详细分析
                        df = calculate_trend_indicators(stock_data)
                        if df is not None:
                            latest = df.iloc[-1]
                            st.write("### 技术指标分析")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("均线系统")
                                st.write(f"MA5: {latest['MA5']:.2f}")
                                st.write(f"MA10: {latest['MA10']:.2f}")
                                st.write(f"MA20: {latest['MA20']:.2f}")
                                st.write(f"MA60: {latest['MA60']:.2f}")
                            
                            with col2:
                                st.write("其他指标")
                                st.write(f"ATR: {latest['ATR']:.2f}")
                                st.write(f"ROC: {latest['ROC']:.2f}")
                                st.write(f"布林带上轨: {latest['BB_upper']:.2f}")
                                st.write(f"布林带下轨: {latest['BB_lower']:.2f}")
            else:
                st.warning("未找到符合条件的股票")

if __name__ == "__main__":
    trend_strategy() 