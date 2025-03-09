import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time as datetime_time
import akshare as ak
import pytz
import time
import os
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.source_util import get_pages


class InvestmentCalculator:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.current_value = initial_capital
        self.history = []  # 用于存储历史记录

    def calculate_return_rate(self):
        return ((self.current_value - self.initial_capital) / self.initial_capital) * 100

    def calculate_profit(self):
        return self.current_value - self.initial_capital

    def update_current_value(self, new_value, date=None):
        self.current_value = new_value
        if date is None:
            date = datetime.now()
        self.history.append({
            "日期": date,
            "价值": new_value,
            "收益率": self.calculate_return_rate(),
            "收益金额": self.calculate_profit()
        })

    def get_summary(self):
        return {
            "初始资金": self.initial_capital,
            "当前价值": self.current_value,
            "收益率": f"{self.calculate_return_rate():.2f}%",
            "收益金额": self.calculate_profit()
        }

def calculate_incremental_values(initial_value, step_percentage=5, max_percentage=95, include_negative=True):
    """计算递增和递减值"""
    results = []
    
    if include_negative:
        # 添加递减值（从-95%到-5%）
        for percentage in range(-max_percentage, 0, step_percentage):
            value = initial_value * (1 + percentage/100)
            results.append({
                "涨跌幅": f"{percentage}%",
                "金额": round(value, 2),
                "收益": round(value - initial_value, 2)
            })
    
    # 添加0%
    results.append({
        "涨跌幅": "0%",
        "金额": round(initial_value, 2),
        "收益": 0.00
    })
    
    # 添加递增值（从5%到95%）
    for percentage in range(step_percentage, max_percentage + step_percentage, step_percentage):
        value = initial_value * (1 + percentage/100)
        results.append({
            "涨跌幅": f"+{percentage}%",
            "金额": round(value, 2),
            "收益": round(value - initial_value, 2)
        })
    
    return results

def is_trading_time():
    """判断当前是否为交易时间"""
    current_time = datetime.now(pytz.timezone('Asia/Shanghai'))
    current_time = current_time.time()
    morning_start = datetime_time(9, 30)
    morning_end = datetime_time(11, 30)
    afternoon_start = datetime_time(13, 0)
    afternoon_end = datetime_time(15, 0)
    
    return ((morning_start <= current_time <= morning_end) or 
            (afternoon_start <= current_time <= afternoon_end))

def get_shanghai_index_close():
    """获取上证指数收盘价"""
    try:
        # 获取当前时间（上海时区）
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(shanghai_tz)
        
        # 判断是否是交易日
        stock_info = ak.tool_trade_date_hist_sina()
        today_str = current_time.strftime('%Y-%m-%d')
        trade_dates = pd.to_datetime(stock_info['trade_date']).dt.strftime('%Y-%m-%d')
        
        if today_str not in trade_dates.values:
            return None, f"今天({today_str})不是交易日"
            
        # 判断是否已经收盘（下午3点后）
        if current_time.hour < 15:
            return None, f"今日({today_str})尚未收盘，当前时间：{current_time.strftime('%H:%M:%S')}"
            
        # 使用新浪财经接口获取上证指数数据
        stock_zh_index_df = ak.index_zh_a_hist(symbol="000001", period="daily", start_date=today_str, end_date=today_str)
        
        # 获取最新数据
        latest_data = stock_zh_index_df.iloc[-1]
        
        # 获取前一日数据用于计算涨跌幅
        prev_day = ak.index_zh_a_hist(
            symbol="000001", 
            period="daily", 
            start_date=(datetime.now() - pd.Timedelta(days=5)).strftime('%Y-%m-%d'),
            end_date=today_str
        )
        
        # 计算涨跌幅
        if len(prev_day) > 1:
            prev_close = prev_day.iloc[-2]['收盘']
            pct_change = ((latest_data['收盘'] - prev_close) / prev_close) * 100
        else:
            pct_change = 0
            
        return {
            "日期": today_str,
            "收盘价": float(latest_data['收盘']),
            "涨跌幅": pct_change,
            "成交量": f"{float(latest_data['成交量'])/10000:.2f}万手",
            "成交额": f"{float(latest_data['成交额'])/100000000:.2f}亿"
        }, "获取成功"
        
    except Exception as e:
        print(f"错误详情: {str(e)}")
        if 'stock_zh_index_df' in locals():
            print("数据列名:", stock_zh_index_df.columns.tolist())
        return None, f"获取数据失败: {str(e)}\n错误类型: {type(e)}"

def get_shanghai_index_realtime():
    """获取上证指数实时数据"""
    try:
        # 获取当前时间（上海时区）
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(shanghai_tz)
        
        # 判断是否是交易日
        stock_info = ak.tool_trade_date_hist_sina()
        today_str = current_time.strftime('%Y-%m-%d')
        trade_dates = pd.to_datetime(stock_info['trade_date']).dt.strftime('%Y-%m-%d')
        
        if today_str not in trade_dates.values:
            return None, f"今天({today_str})不是交易日"
            
        # 判断是否在交易时间内
        current_hour = current_time.hour
        current_minute = current_time.minute
        if not (9 <= current_hour <= 15):
            return None, f"当前不在交易时间内，当前时间：{current_time.strftime('%H:%M:%S')}"
        if current_hour == 9 and current_minute < 30:
            return None, "盘前准备中"
        if current_hour == 11 and current_minute > 30:
            return None, "午间休市"
        if current_hour == 15 and current_minute > 0:
            return None, "已收盘"
            
        # 获取实时数据
        stock_zh_index_df = ak.index_zh_a_hist(symbol="000001", period="daily", start_date=today_str, end_date=today_str)
        latest_data = stock_zh_index_df.iloc[-1]
        
        # 获取前一日数据用于计算涨跌幅
        prev_day = ak.index_zh_a_hist(
            symbol="000001", 
            period="daily", 
            start_date=(datetime.now() - pd.Timedelta(days=5)).strftime('%Y-%m-%d'),
            end_date=today_str
        )
        
        # 计算涨跌幅
        if len(prev_day) > 1:
            prev_close = prev_day.iloc[-2]['收盘']
            pct_change = ((latest_data['收盘'] - prev_close) / prev_close) * 100
        else:
            pct_change = 0
            
        return {
            "日期": today_str,
            "时间": current_time.strftime('%H:%M:%S'),
            "收盘价": float(latest_data['收盘']),
            "涨跌幅": pct_change,
            "成交量": f"{float(latest_data['成交量'])/10000:.2f}万手",
            "成交额": f"{float(latest_data['成交额'])/100000000:.2f}亿"
        }, "获取成功"
        
    except Exception as e:
        print(f"错误详情: {str(e)}")
        return None, f"获取数据失败: {str(e)}\n错误类型: {type(e)}"

def get_index_decline_records(start_date="2025-01-01"):
    try:
        # 获取当前时间
        current_time = datetime.now(pytz.timezone('Asia/Shanghai'))
        end_date = current_time.strftime('%Y-%m-%d')
        
        # 获取历史数据
        hist_data = ak.index_zh_a_hist(
            symbol="000001", 
            period="daily", 
            start_date=start_date,
            end_date=end_date
        )
        
        # 打印原始数据
        with st.expander("原始数据"):
            st.write("原始数据形状:", hist_data.shape)
            st.write("原始数据列名:", hist_data.columns.tolist())
            st.dataframe(hist_data, height=400, use_container_width=True)
        
        # 确保日期列是datetime类型
        hist_data['日期'] = pd.to_datetime(hist_data['日期'])
        
        # 计算涨跌幅
        hist_data['前日收盘'] = hist_data['收盘'].shift(1)
        hist_data['涨跌幅'] = ((hist_data['收盘'] - hist_data['前日收盘']) / hist_data['前日收盘'] * 100)
        
        # 打印涨跌幅计算后的数据
        with st.expander("涨跌幅计算后的数据"):
            st.write("涨跌幅数据形状:", hist_data.shape)
            st.write("涨跌幅范围:", f"最小值: {hist_data['涨跌幅'].min():.2f}%, 最大值: {hist_data['涨跌幅'].max():.2f}%")
            styled_hist = hist_data.style.format({
                '涨跌幅': '{:.2f}%',
                '收盘': '{:.2f}',
                '成交量': '{:.2f}',
                '成交额': '{:.2f}'
            }).applymap(
                lambda x: 'color: red' if x > 0 else 'color: green' if x < 0 else '',
                subset=['涨跌幅']
            )
            st.dataframe(styled_hist, height=400, use_container_width=True)
        
        # 筛选涨跌幅绝对值大于1%的记录
        volatility_records = hist_data[abs(hist_data['涨跌幅']) > 1].copy()
        
        # 打印筛选后的数据
        with st.expander("筛选后的数据（涨跌幅绝对值>1%）"):
            st.write("筛选后数据形状:", volatility_records.shape)
            styled_volatility = volatility_records.style.format({
                '涨跌幅': '{:.2f}%',
                '收盘': '{:.2f}',
                '成交量': '{:.2f}',
                '成交额': '{:.2f}'
            }).applymap(
                lambda x: 'color: red' if x > 0 else 'color: green' if x < 0 else '',
                subset=['涨跌幅']
            )
            st.dataframe(styled_volatility, height=400, use_container_width=True)
        
        # 格式化数据
        volatility_records = volatility_records.reset_index(drop=True)
        volatility_records['日期'] = volatility_records['日期'].dt.strftime('%Y-%m-%d')
        volatility_records['涨跌幅'] = volatility_records['涨跌幅'].round(2)
        volatility_records['收盘价'] = volatility_records['收盘'].round(2)
        volatility_records['成交量'] = (volatility_records['成交量'] / 10000).round(2)
        volatility_records['成交额'] = (volatility_records['成交额'] / 100000000).round(2)
        
        # 选择需要显示的列并按日期降序排序
        result = volatility_records[[
            '日期', '收盘价', '涨跌幅', '成交量', '成交额'
        ]].rename(columns={
            '成交量': '成交量(万手)',
            '成交额': '成交额(亿)'
        }).sort_values('日期', ascending=False)
        
        if len(result) > 0:
            stats = {
                'count': len(result),
                'up_count': len(result[result['涨跌幅'] > 1]),
                'down_count': len(result[result['涨跌幅'] < -1]),
                'max_up': result['涨跌幅'].max(),
                'max_down': result['涨跌幅'].min(),
                'recent': result.iloc[0]['日期'],
                'max_volume': result['成交量(万手)'].max()
            }
            return result, stats
        
        return None, None
        
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        return None, None
def get_chinext_decline_records(start_date="2025-01-01"):
    try:
        # 获取当前时间
        current_time = datetime.now(pytz.timezone('Asia/Shanghai'))
        end_date = current_time.strftime('%Y-%m-%d')
        
        # 获取历史数据
        hist_data = ak.index_zh_a_hist(
            symbol="399006", # 创业板指数代码
            period="daily", 
            start_date=start_date,
            end_date=end_date
        )
        
        # 打印原始数据
        with st.expander("创业板指数原始数据"):
            st.write("原始数据形状:", hist_data.shape)
            st.write("原始数据列名:", hist_data.columns.tolist())
            st.dataframe(hist_data, height=400, use_container_width=True)
        
        # 确保日期列是datetime类型
        hist_data['日期'] = pd.to_datetime(hist_data['日期'])
        
        # 计算涨跌幅
        hist_data['前日收盘'] = hist_data['收盘'].shift(1)
        hist_data['涨跌幅'] = ((hist_data['收盘'] - hist_data['前日收盘']) / hist_data['前日收盘'] * 100)
        
        # 打印涨跌幅计算后的数据
        with st.expander("创业板指数涨跌幅计算后的数据"):
            st.write("涨跌幅数据形状:", hist_data.shape)
            st.write("涨跌幅范围:", f"最小值: {hist_data['涨跌幅'].min():.2f}%, 最大值: {hist_data['涨跌幅'].max():.2f}%")
            styled_hist = hist_data.style.format({
                '涨跌幅': '{:.2f}%',
                '收盘': '{:.2f}',
                '成交量': '{:.2f}',
                '成交额': '{:.2f}'
            }).applymap(
                lambda x: 'color: red' if x > 0 else 'color: green' if x < 0 else '',
                subset=['涨跌幅']
            )
            st.dataframe(styled_hist, height=400, use_container_width=True)
        
        # 筛选涨跌幅绝对值大于1%的记录
        volatility_records = hist_data[abs(hist_data['涨跌幅']) > 1].copy()
        
        # 打印筛选后的数据
        with st.expander("创业板指数筛选后的数据（涨跌幅绝对值>1%）"):
            st.write("筛选后数据形状:", volatility_records.shape)
            styled_volatility = volatility_records.style.format({
                '涨跌幅': '{:.2f}%',
                '收盘': '{:.2f}',
                '成交量': '{:.2f}',
                '成交额': '{:.2f}'
            }).applymap(
                lambda x: 'color: red' if x > 0 else 'color: green' if x < 0 else '',
                subset=['涨跌幅']
            )
            st.dataframe(styled_volatility, height=400, use_container_width=True)
        
        # 格式化数据
        volatility_records = volatility_records.reset_index(drop=True)
        volatility_records['日期'] = volatility_records['日期'].dt.strftime('%Y-%m-%d')
        volatility_records['涨跌幅'] = volatility_records['涨跌幅'].round(2)
        volatility_records['收盘价'] = volatility_records['收盘'].round(2)
        volatility_records['成交量'] = (volatility_records['成交量'] / 10000).round(2)
        volatility_records['成交额'] = (volatility_records['成交额'] / 100000000).round(2)
        
        # 选择需要显示的列并按日期降序排序
        result = volatility_records[[
            '日期', '收盘价', '涨跌幅', '成交量', '成交额'
        ]].rename(columns={
            '成交量': '成交量(万手)',
            '成交额': '成交额(亿)'
        }).sort_values('日期', ascending=False)
        
        if len(result) > 0:
            stats = {
                'count': len(result),
                'up_count': len(result[result['涨跌幅'] > 1]),
                'down_count': len(result[result['涨跌幅'] < -1]),
                'max_up': result['涨跌幅'].max(),
                'max_down': result['涨跌幅'].min(),
                'recent': result.iloc[0]['日期'],
                'max_volume': result['成交量(万手)'].max()
            }
            return result, stats
        
        return None, None
        
    except Exception as e:
        st.error(f"获取创业板指数数据失败: {str(e)}")
        return None, None

def write_changelog():
    """创建版本记录文件"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建 CHANGELOG.md 的完整路径
    changelog_path = os.path.join(current_dir, 'CHANGELOG.md')
    
    changelog_content = """# 投资收益计算器更新日志

## Version 1.0.0 (2024-03-21)

### 主要功能
1. 投资收益计算器基础功能
2. 上证指数实时监控
3. 上证指数跌幅统计（2025-01-01至今）
4. 数据调试查看功能

### 关键代码结构

stock/investment_app.py
├── class InvestmentCalculator # 投资计算器类
├── def calculate_incremental_values() # 计算涨跌幅参考值
├── def get_shanghai_index_close() # 获取上证指数收盘数据
├── def get_shanghai_index_realtime() # 获取上证指数实时数据
├── def get_index_decline_records() # 获取上证指数跌幅统计
└── def main() # 主函数
├── 上证指数跌幅统计模块
├── 实时监控模块
├── 收益计算模块
└── 涨跌幅参考表


### 功能特点

#### 1. 上证指数跌幅统计
- 统计2025年起跌幅超过1%的交易日
- 显示详细数据表格
- 提供统计信息（天数、平均跌幅、最大跌幅等）
- 包含数据调试查看功能

#### 2. 实时监控功能
- 支持开启/关闭实时监控
- 可调整刷新间隔（5-60秒）
- 显示实时价格、涨跌幅、成交量等
- 涨跌幅颜色区分（红涨绿跌）

#### 3. 收益计算功能
- 支持输入初始资金和当前价值
- 自动计算收益率和收益金额
- 提供收益率仪表盘
- 显示涨跌幅参考表

#### 4. 数据展示优化
- 使用 Streamlit 组件优化显示
- 表格样式美化
- 可展开的数据详情查看
- 清晰的布局结构

### 依赖库
python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz
import akshare as ak
import time

### 使用说明
1. 在终端运行：`streamlit run stock/investment_app.py`
2. 通过左侧边栏控制实时监控
3. 点击"查看数据详情"可以查看完整数据信息
4. 输入初始资金和当前价值计算收益

### 后续可优化方向
1. 添加更多技术指标
2. 优化数据刷新机制
3. 增加历史数据分析功能
4. 添加数据导出功能

"""

    
    try:
        # 写入 CHANGELOG.md 文件
        with open(changelog_path, 'w', encoding='utf-8') as f:
            f.write(changelog_content)
        print(f"版本记录已成功写入: {changelog_path}")
        return True, changelog_path
    except Exception as e:
        print(f"写入版本记录失败: {str(e)}")
        return False, str(e)

def main():
    st.set_page_config(page_title="投资收益计算器", layout="wide")
    st.title("📈 投资收益计算器")

    # 显示跌幅统计
    st.header("上证指数跌幅统计")
    st.caption("2025年1月1日至今跌幅超过1%的交易日")
    
    volatility_data, stats = get_index_decline_records()
    if volatility_data is not None and not volatility_data.empty and stats is not None:
        # 使用 styler 来设置表格样式，根据涨跌幅设置不同颜色
        styled_df = volatility_data.style.format({
            '涨跌幅': '{:.2f}%',
            '收盘价': '{:.2f}',
            '成交量(万手)': '{:.2f}',
            '成交额(亿)': '{:.2f}'
        }).applymap(
            lambda x: 'color: red' if x > 1 else 'color: green' if x < -1 else '',
            subset=['涨跌幅']
        )
        
        # 显示统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("上涨天数(>1%)", f"{stats['up_count']}天")
        with col2:
            st.metric("下跌天数(<-1%)", f"{stats['down_count']}天")
        with col3:
            st.metric("波动天数总计", f"{stats['count']}天")
        
        # 显示数据表格
        st.dataframe(
            styled_df,
            height=400,
            use_container_width=True
        )
        
        # 显示详细统计信息
        st.info(f"""
        详细统计：
        - 统计区间：2025-01-01 至 {datetime.now().strftime('%Y-%m-%d')}
        - 最大涨幅：{stats['max_up']:.2f}%
        - 最大跌幅：{stats['max_down']:.2f}%
        - 最近波动日期：{stats['recent']}
        - 期间最大成交量：{stats['max_volume']:.2f}万手
        """)
    else:
        st.warning("暂无涨跌幅超过1%的记录")

     # 显示跌幅统计
        # 显示创业板指数跌幅统计
    st.header("创业板指数跌幅统计")
    st.caption("2025年1月1日至今跌幅超过1%的交易日")
    
    chinext_data, chinext_stats = get_chinext_decline_records()
    if chinext_data is not None and not chinext_data.empty and chinext_stats is not None:
        # 使用 styler 来设置表格样式，根据涨跌幅设置不同颜色
        styled_df = chinext_data.style.format({
            '涨跌幅': '{:.2f}%',
            '收盘价': '{:.2f}',
            '成交量(万手)': '{:.2f}',
            '成交额(亿)': '{:.2f}'
        }).applymap(
            lambda x: 'color: red' if x > 1 else 'color: green' if x < -1 else '',
            subset=['涨跌幅']
        )
        
        # 显示统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("上涨天数(>1%)", f"{chinext_stats['up_count']}天")
        with col2:
            st.metric("下跌天数(<-1%)", f"{chinext_stats['down_count']}天")
        with col3:
            st.metric("波动天数总计", f"{chinext_stats['count']}天")
        
        # 显示数据表格
        st.dataframe(
            styled_df,
            height=400,
            use_container_width=True
        )
        
        # 显示详细统计信息
        st.info(f"""
        详细统计：
        - 统计区间：2025-01-01 至 {datetime.now().strftime('%Y-%m-%d')}
        - 最大涨幅：{chinext_stats['max_up']:.2f}%
        - 最大跌幅：{chinext_stats['max_down']:.2f}%
        - 最近波动日期：{chinext_stats['recent']}
        - 期间最大成交量：{chinext_stats['max_volume']:.2f}万手
        """)
    else:
        st.warning("暂无创业板指数涨跌幅超过1%的记录")
    # 添加上证指数信息显示
    st.sidebar.header("上证指数信息")
    
    # 添加监听控制
    monitor_active = st.sidebar.checkbox("开启实时监控", value=False)
    if monitor_active:
        refresh_interval = st.sidebar.slider("刷新间隔(秒)", 5, 60, 10)
    
    # 创建一个占位符用于更新数据
    index_container = st.sidebar.empty()
    
    while monitor_active:
        with index_container.container():
            index_data, message = get_shanghai_index_realtime()
            
            if index_data:
                st.success(f"更新时间: {index_data['日期']} {index_data['时间']}")
                col1, col2 = st.columns(2)
                
                # 获取涨跌幅的值和颜色
                pct_change = index_data['涨跌幅']
                pct_color = "red" if pct_change > 0 else "green" if pct_change < 0 else "gray"
                
                with col1:
                    st.metric("收盘价", f"{index_data['收盘价']:.2f}")
                    st.metric("成交量", index_data['成交量'])
                with col2:
                    st.write("涨跌幅")
                    st.markdown(f"<p style='color: {pct_color}; font-size: 1.2em;'>{pct_change:+.2f}%</p>", 
                              unsafe_allow_html=True)
                    st.metric("成交额", index_data['成交额'])
            else:
                st.warning(message)
            
        time.sleep(refresh_interval)
    
    # 如果没有开启监控，显示普通收盘数据
    if not monitor_active:
        index_data, message = get_shanghai_index_close()
        
        if index_data:
            st.sidebar.success(f"更新时间: {index_data['日期']}")
            col1, col2 = st.sidebar.columns(2)
            
            pct_change = index_data['涨跌幅']
            pct_color = "red" if pct_change > 0 else "green" if pct_change < 0 else "gray"
            
            with col1:
                st.metric("收盘价", f"{index_data['收盘价']:.2f}")
                st.metric("成交量", index_data['成交量'])
            with col2:
                st.write("涨跌幅")
                st.markdown(f"<p style='color: {pct_color}; font-size: 1.2em;'>{pct_change:+.2f}%</p>", 
                          unsafe_allow_html=True)
                st.metric("成交额", index_data['成交额'])
        else:
            st.sidebar.warning(message)

        # 侧边栏输入
        with st.sidebar:
            st.header("输入参数")
            initial_capital = st.number_input("初始资金", min_value=0.0, value=10000.0, step=1000.0)
            current_value = st.number_input("当前价值", min_value=0.0, value=10000.0, step=1000.0)
            
            # 添加分隔线
            st.markdown("---")
            
            # 添加版本记录按钮
            if st.button("生成版本记录"):
                success, message = write_changelog()
                if success:
                    st.success(f"版本记录已成功生成：\n{message}")
                else:
                    st.error(f"生成版本记录失败：{message}")
            

        # 创建计算器实例
        calculator = InvestmentCalculator(initial_capital)
        calculator.update_current_value(round(current_value, 2))
        summary = calculator.get_summary()

        # 显示主要指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("初始资金", f"¥{summary['初始资金']:,.2f}")
        with col2:
            st.metric("当前价值", f"¥{summary['当前价值']:,.2f}")
        with col3:
            st.metric("收益率", f"{float(summary['收益率'].rstrip('%')):.2f}%")
        with col4:
            st.metric("收益金额", f"¥{summary['收益金额']:,.2f}")

        # 第一行：收益率仪表盘
        st.header("收益率仪表盘")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = calculator.calculate_return_rate(),
            title = {'text': "当前收益率"},
            gauge = {
                'axis': {'range': [-50, 50]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-50, 0], 'color': "lightcoral"},
                    {'range': [0, 50], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # 第二行：涨跌幅参考表
        st.header("涨跌幅参考表")
        
        # 获取所有涨跌幅数据
        all_values = calculate_incremental_values(initial_capital)
        df = pd.DataFrame(all_values)
        
        # 将数据分成四组
        negative_large = df[df['涨跌幅'].apply(lambda x: '-' in x and int(x.strip('%').strip('-')) > 50)]
        negative_small = df[df['涨跌幅'].apply(lambda x: '-' in x and int(x.strip('%').strip('-')) <= 50)]
        zero = df[df['涨跌幅'] == '0%']
        positive_small = df[df['涨跌幅'].apply(lambda x: '+' in x and int(x.strip('%').strip('+')) <= 50)]
        positive_large = df[df['涨跌幅'].apply(lambda x: '+' in x and int(x.strip('%').strip('+')) > 50)]

        # 创建四列布局显示涨跌幅表格
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("大幅下跌")
            st.caption("-95% ~ -55%")
            st.dataframe(
                negative_large.style.format({
                    "金额": "¥{:,.2f}",
                    "收益": "¥{:,.2f}"
                }).applymap(lambda x: 'color: red', subset=['涨跌幅', '收益']),
                height=300
            )

        with col2:
            st.subheader("小幅下跌")
            st.caption("-50% ~ -5%")
            st.dataframe(
                negative_small.style.format({
                    "金额": "¥{:,.2f}",
                    "收益": "¥{:,.2f}"
                }).applymap(lambda x: 'color: red', subset=['涨跌幅', '收益']),
                height=300
            )

        with col3:
            st.subheader("小幅上涨")
            st.caption("0% ~ +50%")
            combined_small = pd.concat([zero, positive_small])
            st.dataframe(
                combined_small.style.format({
                    "金额": "¥{:,.2f}",
                    "收益": "¥{:,.2f}"
                }).applymap(
                    lambda x: 'color: green' if '+' in str(x) else 'color: black',
                    subset=['涨跌幅']
                ).applymap(
                    lambda x: 'color: green' if x > 0 else 'color: black',
                    subset=['收益']
                ),
                height=300
            )

        with col4:
            st.subheader("大幅上涨")
            st.caption("+55% ~ +95%")
            st.dataframe(
                positive_large.style.format({
                    "金额": "¥{:,.2f}",
                    "收益": "¥{:,.2f}"
                }).applymap(lambda x: 'color: green', subset=['涨跌幅', '收益']),
                height=300
            )

        # 使用说明
        st.markdown("""
        ### 使用说明
        1. 在左侧边栏输入您的初始投资金额
        2. 输入当前投资价值
        3. 系统会自动计算收益率和收益金额
        4. 参考表分为四个区域：
           - 大幅下跌区（-95% ~ -55%）
           - 小幅下跌区（-50% ~ -5%）
           - 小幅上涨区（0% ~ +50%）
           - 大幅上涨区（+55% ~ +95%）
        - 红色表示亏损数值
        - 绿色表示盈利数值
        """)

if __name__ == "__main__":
    main()
