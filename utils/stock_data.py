import akshare as ak
import pandas as pd
from .technical_analysis import check_buy_signal
import time
import streamlit as st
import concurrent.futures
import threading
from queue import Queue
from requests.exceptions import RequestException, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError, IncompleteRead
import random
from functools import wraps
from threading import Lock
from tqdm import tqdm
import urllib3
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import asyncio
import os
import json
from datetime import datetime, timedelta
import pickle
from pathlib import Path

# 创建一个线程安全的队列来存储进度信息
progress_queue = Queue()
selected_stocks_queue = Queue()

# 创建一个锁用于同步进度更新
progress_lock = Lock()

# 配置 urllib3
urllib3.disable_warnings()
http = urllib3.PoolManager(retries=urllib3.Retry(3, backoff_factor=0.5))

# 创建缓存目录
CACHE_DIR = Path("cache/stock_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class StockDataCache:
    """
    股票数据缓存管理类
    
    用于管理股票数据的本地缓存，减少重复请求，提高性能。
    
    属性：
        cache_days (int): 缓存有效期（天）
        
    示例：
        >>> cache = StockDataCache(cache_days=1)
        >>> data = cache.get_cached_data("000001")
    """
    
    def __init__(self, cache_days=1):
        """
        初始化缓存管理器
        
        参数：
            cache_days (int): 缓存数据的有效期，单位为天
        """
        self.cache_days = cache_days
        
    def get_cache_path(self, stock_code):
        """
        获取股票数据的缓存文件路径
        
        参数：
            stock_code (str): 股票代码
            
        返回：
            Path: 缓存文件的路径
        """
        return CACHE_DIR / f"{stock_code}.pkl"
        
    def is_cache_valid(self, cache_path):
        if not cache_path.exists():
            return False
        
        # 检查文件修改时间
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return (datetime.now() - mtime).days < self.cache_days
        
    def get_cached_data(self, stock_code):
        cache_path = self.get_cache_path(stock_code)
        if self.is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
        
    def save_to_cache(self, stock_code, data):
        if data is not None and not data.empty:
            cache_path = self.get_cache_path(stock_code)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                st.warning(f"缓存数据保存失败: {str(e)}")

# 创建缓存实例
stock_cache = StockDataCache()

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """自定义重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep_time = (backoff_in_seconds * 2 ** x +
                                random.uniform(0, 1))
                    time.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator

@retry_with_backoff(retries=3, backoff_in_seconds=2)
def _get_stock_info():
    """获取A股股票列表"""
    time.sleep(random.uniform(0.5, 2))
    try:
        df = ak.stock_info_a_code_name()
        if len(df.columns) >= 2:
            df = df.iloc[:, :2].copy()
            df.columns = ['代码', '名称']
        return df
    except Exception as e:
        st.error(f"获取股票信息时出错: {str(e)}")
        raise

@retry_with_backoff(retries=3, backoff_in_seconds=2)
def _get_realtime_quotes():
    """
    获取实时行情数据
    
    内部函数，使用多个数据源备份机制获取实时行情。
    
    返回：
        DataFrame: 包含所有A股实时行情数据
        
    注意：
        - 包含自动重试机制
        - 在请求失败时会自动切换数据源
    """
    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def fetch_quotes():
        try:
            # 尝试不同的数据源
            try:
                # 尝试使用东方财富接口
                df = ak.stock_zh_a_spot_em()
            except Exception as e1:
                st.warning(f"东方财富接口获取失败: {str(e1)}")
                try:
                    # 尝试使用新浪接口
                    df = ak.stock_zh_a_spot()
                except Exception as e2:
                    st.warning(f"新浪接口获取失败: {str(e2)}")
                    # 最后尝试腾讯接口
                    df = ak.stock_zh_a_spot_qq()
            
            if df is not None and not df.empty:
                # 统一列名
                rename_dict = {
                    '代码': '代码',
                    'code': '代码',
                    '名称': '名称',
                    'name': '名称',
                    '最新价': '最新价',
                    'price': '最新价',
                    '涨跌幅': '涨跌幅',
                    'change_percent': '涨跌幅',
                    '成交量': '成交量',
                    'volume': '成交量',
                    '成交额': '成交额',
                    'amount': '成交额',
                    '换手率': '换手率',
                    'turnover': '换手率'
                }
                
                # 只重命名存在的列
                rename_cols = {k: v for k, v in rename_dict.items() if k in df.columns}
                df = df.rename(columns=rename_cols)
                
                # 确保数值类型正确
                numeric_cols = ['最新价', '涨跌幅', '成交量', '成交额', '换手率']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            
            return None
            
        except Exception as e:
            st.error(f"获取实时行情时出错: {str(e)}")
            raise
            
    return fetch_quotes()

def get_stock_data(stock_code, days=30):
    """
    获取单只股票的历史数据（同步接口）
    
    这是一个向后兼容的同步接口，内部使用异步实现。
    
    参数：
        stock_code (str): 股票代码
        days (int): 需要获取的天数
        
    返回：
        DataFrame: 包含股票历史数据的DataFrame，如果获取失败返回None
        
    DataFrame列：
        - 日期 (datetime64[ns]): 交易日期
        - 开盘 (float): 开盘价
        - 收盘 (float): 收盘价
        - 最高 (float): 最高价
        - 最低 (float): 最低价
        - 成交量 (float): 成交量
        - 成交额 (float): 成交额
        - 换手率 (float): 换手率
        - 涨跌幅 (float): 涨跌幅
        
    示例：
        >>> df = get_stock_data('000001', days=30)
        >>> if df is not None:
        ...     print(df.head())
    """
    try:
        # 先检查缓存
        cached_data = stock_cache.get_cached_data(stock_code)
        if cached_data is not None:
            return cached_data.tail(days)
        
        # 确保股票代码是字符串并去除空格
        stock_code = str(stock_code).strip()
        
        # 设置日期范围
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=days*2)
        
        # 根据不同市场使用不同的数据源
        if stock_code.startswith('6'):  # 上证
            try:
                df = ak.stock_zh_a_hist(  # 使用东方财富接口
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"
                )
            except Exception as e:
                st.warning(f"东方财富接口失败，尝试备用接口: {str(e)}")
                df = ak.stock_zh_a_daily(  # 备用新浪接口
                    symbol=f"sh{stock_code}",
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"
                )
        elif stock_code.startswith('8'):  # 北交所
            try:
                df = ak.stock_bj_a_hist(  # 使用北交所专用接口
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"
                )
            except Exception as e:
                st.warning(f"北交所接口失败，尝试备用接口: {str(e)}")
                df = ak.stock_zh_a_hist(  # 备用东方财富接口
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"
                )
        else:  # 深证和创业板
            try:
                df = ak.stock_zh_a_hist(  # 使用东方财富接口
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"
                )
            except Exception as e:
                st.warning(f"东方财富接口失败，尝试备用接口: {str(e)}")
                df = ak.stock_zh_a_daily(  # 备用新浪接口
                    symbol=f"sz{stock_code}",
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"
                )
        
        if df is None or df.empty:
            st.warning(f"无法获取股票 {stock_code} 的数据")
            return None
            
        # 确保列名正确
        column_mappings = {
            '日期': '日期',
            'date': '日期',
            '开盘': '开盘',
            'open': '开盘',
            '收盘': '收盘',
            'close': '收盘',
            '最高': '最高',
            'high': '最高',
            '最低': '最低',
            'low': '最低',
            '成交量': '成交量',
            'volume': '成交量',
            '成交额': '成交额',
            'amount': '成交额',
            '振幅': '振幅',
            '涨跌幅': '涨跌幅',
            'pct_chg': '涨跌幅',
            '涨跌额': '涨跌额',
            '换手率': '换手率',
            'turnover': '换手率'
        }
        
        # 只重命名存在的列
        rename_dict = {k: v for k, v in column_mappings.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        # 确保所有数值列都是数值类型
        numeric_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率', '涨跌幅']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 确保日期列是datetime类型
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 按日期排序并获取最新数据
        df = df.sort_values('日期').tail(days)
        
        # 保存到缓存
        stock_cache.save_to_cache(stock_code, df)
        
        return df
        
    except Exception as e:
        st.error(f"获取股票 {stock_code} 数据时发生错误: {str(e)}")
        import traceback
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None

def should_retry(exception):
    """判断是否需要重试的条件"""
    retry_exceptions = (
        ConnectionError,
        TimeoutError,
        KeyError,
        ValueError,
        Exception
    )
    return isinstance(exception, retry_exceptions)

@retry_with_backoff(retries=5, backoff_in_seconds=2)
def get_stock_list_with_retry():
    """获取股票列表（带重试机制）"""
    try:
        # 随机延时0.5-2秒
        time.sleep(random.uniform(0.5, 2))
        return ak.stock_info_a_code_name()
    except Exception as e:
        print(f"获取股票列表时发生错误: {str(e)}")
        raise

@retry_with_backoff(retries=5, backoff_in_seconds=2)
def get_realtime_quotes_with_retry():
    """获取实时行情（带重试机制）"""
    try:
        # 随机延时0.5-2秒
        time.sleep(random.uniform(0.5, 2))
        return ak.stock_zh_a_spot_em()
    except Exception as e:
        print(f"获取实时行情时发生错误: {str(e)}")
        raise

def process_stock(stock, latest_quotes, progress_bar, total_stocks):
    """处理单个股票的函数"""
    try:
        stock_data = get_stock_data(stock['代码'])
        
        if stock_data is not None and not stock_data.empty:
            is_buy, reason = check_buy_signal(stock_data)
            
            if is_buy:
                stock_quote = latest_quotes[latest_quotes['代码'] == stock['代码']]
                
                if not stock_quote.empty:
                    return {
                        '代码': stock['代码'],
                        '名称': stock['名称'],
                        '板块': stock['板块'],
                        '现价': float(stock_quote['最新价'].iloc[0]) if '最新价' in stock_quote else 0.0,
                        '涨跌幅': float(stock_quote['涨跌幅'].iloc[0]) if '涨跌幅' in stock_quote else 0.0,
                        '换手率': float(stock_quote['换手率'].iloc[0]) if '换手率' in stock_quote else 0.0,
                        '成交额': float(stock_quote['成交额'].iloc[0]) if '成交额' in stock_quote else 0.0,
                        '选股理由': reason
                    }
    except Exception as e:
        st.error(f"处理股票 {stock['代码']} 时发生错误: {str(e)}")
    return None

def update_progress():
    """更新进度条的函数"""
    progress_bar = st.progress(0.0)
    while True:
        progress = progress_queue.get()
        if progress == -1:  # 结束信号
            progress_bar.progress(1.0)
            break
        progress_bar.progress(progress)

def get_board(stock_code):
    """获取股票所属板块"""
    stock_code = str(stock_code)
    if stock_code.startswith('60'):
        return '上证主板'
    elif stock_code.startswith('00'):
        return '深证主板'
    elif stock_code.startswith('30'):
        return '创业板'
    elif stock_code.startswith('68'):
        return '科创板'
    elif stock_code.startswith('83'):
        return '北交所'
    else:
        return '其他'

def get_stock_list(selected_boards=None):
    """获取A股股票列表并进行策略筛选"""
    try:
        start_time = time.time()  # 记录开始时间
        
        # 获取所有A股信息
        stock_info = _get_stock_info()
        if stock_info is None or stock_info.empty:
            st.error("无法获取股票列表")
            return pd.DataFrame()
            
        # 添加板块信息
        stock_info = stock_info.copy()
        stock_info['板块'] = stock_info['代码'].apply(lambda x: get_board(x))
        
        # 显示各板块数量
        board_counts = stock_info['板块'].value_counts()
        st.write("### 各板块股票数量")
        st.write(board_counts)
        
        # 排除ST股票
        non_st_stocks = stock_info[~stock_info['名称'].str.contains('ST|退')].copy()
        st.write(f"排除ST后股票数量: {len(non_st_stocks)}")
        
        # 按选择的板块筛选股票
        if selected_boards:
            filtered_stocks = non_st_stocks[non_st_stocks['板块'].isin(selected_boards)].copy()
        else:
            filtered_stocks = non_st_stocks.copy()
            
        st.write(f"选中板块的股票数量: {len(filtered_stocks)}")
        
        # 获取实时行情数据
        latest_quotes = _get_realtime_quotes()
        
        # 存储筛选结果
        selected_stocks = []
        total_stocks = len(filtered_stocks)
        
        # 将 DataFrame 转换为字典列表，提高性能
        stock_records = filtered_stocks.to_dict('records')
        
        # 创建进度条
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        
        # 使用线程池处理股票
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(
                    process_stock,
                    stock,
                    latest_quotes,
                    progress_bar,
                    total_stocks
                ): stock for stock in stock_records
            }
            
            completed = 0
            
            # 使用 as_completed 处理完成的任务，避免同步等待
            for future in concurrent.futures.as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    result = future.result()
                    if result:
                        selected_stocks.append(result)
                except Exception as e:
                    st.error(f"处理股票 {stock['代码']} 时发生错误: {str(e)}")
                
                # 更新进度和时间统计
                completed += 1
                progress = min(1.0, completed / total_stocks)
                progress_bar.progress(progress)
                
                # 计算处理速度和预估剩余时间
                elapsed_time = time.time() - start_time
                avg_time_per_stock = elapsed_time / completed
                remaining_stocks = total_stocks - completed
                estimated_time = remaining_stocks * avg_time_per_stock
                
                # 格式化时间显示
                elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
                estimated_str = time.strftime("%M:%S", time.gmtime(estimated_time))
                
                # 计算实时处理速度（每秒处理的股票数）
                current_speed = completed / elapsed_time
                
                progress_text.text(
                    f"处理进度: {completed}/{total_stocks} | "
                    f"已用时间: {elapsed_str} | "
                    f"预计剩余: {estimated_str} | "
                    f"实时速度: {current_speed:.1f} 只/秒"
                )
        
        # 计算总耗时
        total_time = time.time() - start_time
        total_time_str = time.strftime("%M:%S", time.gmtime(total_time))
        
        # 清除进度显示
        progress_text.empty()
        progress_bar.empty()
        
        if not selected_stocks:
            st.warning("未找到符合条件的股票")
            return pd.DataFrame()
            
        # 创建结果DataFrame
        result_df = pd.DataFrame(selected_stocks)
        
        # 设置列顺序
        columns = ['代码', '名称', '板块', '现价', '涨跌幅', '换手率', '成交额', '选股理由']
        result_df = result_df.reindex(columns=columns)
        
        # 对结果进行排序
        result_df = result_df.sort_values(['板块', '涨跌幅'], ascending=[True, False])
        
        st.write("### 各板块选股结果")
        st.write(result_df['板块'].value_counts())
        st.write(f"分析完成，共找到 {len(result_df)} 只符合条件的股票")
        st.write(f"总耗时: {total_time_str} | 平均速度: {total_stocks/total_time:.1f} 只/秒")
        
        return result_df
        
    except Exception as e:
        st.error(f"选股过程发生错误: {str(e)}")
        st.write("错误详情:", e)
        return pd.DataFrame()

def check_buy_signal(stock_data):
    """检查是否有买入信号"""
    try:
        if stock_data is None or len(stock_data) < 5:
            return False, "数据不足"
            
        # 计算MA5
        stock_data['MA5'] = stock_data['收盘'].rolling(window=5).mean()
        
        # 获取最新的MA5斜率
        latest_ma5 = stock_data['MA5'].iloc[-5:].values
        slope = (latest_ma5[-1] - latest_ma5[0]) / 5
        
        # 检查是否满足买入条件
        if slope > 0:
            return True, f"MA5斜率为正 ({slope:.4f})"
        
        return False, f"MA5斜率为负 ({slope:.4f})"
        
    except Exception as e:
        st.error(f"检查买入信号时发生错误: {str(e)}")
        return False, f"检查失败: {str(e)}"

def calculate_ma5_slope(stock_data):
    """计算MA5斜率"""
    try:
        if stock_data is None or len(stock_data) < 5:
            return 0
            
        # 确保收盘价是数值类型
        stock_data['收盘'] = pd.to_numeric(stock_data['收盘'], errors='coerce')
        
        # 计算MA5
        stock_data['MA5'] = stock_data['收盘'].rolling(window=5).mean()
        
        # 获取最新的MA5斜率
        latest_ma5 = stock_data['MA5'].tail(5).values
        
        # 使用线性回归计算斜率
        x = np.arange(len(latest_ma5))
        slope, _ = np.polyfit(x, latest_ma5, 1)
        
        return slope
        
    except Exception as e:
        st.error(f"计算MA5斜率时发生错误: {str(e)}")
        return 0

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

async def fetch_stock_data_async(stock_code, session):
    """异步获取股票数据"""
    try:
        # 先检查缓存
        cached_data = stock_cache.get_cached_data(stock_code)
        if cached_data is not None:
            return stock_code, cached_data
            
        # 构建正确的股票代码格式
        if stock_code.startswith('6'):
            market = 'sh'
        elif stock_code.startswith(('0', '3')):
            market = 'sz'
        elif stock_code.startswith('8'):
            market = 'bj'
        else:
            market = 'sz'
            
        full_code = f"{market}{stock_code}"
        
        # 设置日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 获取更多历史数据
        
        # 使用 aiohttp 异步请求数据
        async with session.get(
            f"http://api.finance.ifeng.com/akdaily",
            params={
                "code": full_code,
                "type": "last"
            }
        ) as response:
            data = await response.json()
            
            if "record" in data:
                df = pd.DataFrame(data["record"])
                df.columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                
                # 数据处理
                df['日期'] = pd.to_datetime(df['日期'])
                numeric_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率', '涨跌幅']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 保存到缓存
                stock_cache.save_to_cache(stock_code, df)
                
                return stock_code, df
                
        return stock_code, None
        
    except Exception as e:
        st.error(f"获取股票 {stock_code} 数据时发生错误: {str(e)}")
        return stock_code, None

async def fetch_all_stocks_data(stock_codes):
    """异步获取所有股票数据"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for code in stock_codes:
            task = asyncio.create_task(fetch_stock_data_async(code, session))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        return {code: data for code, data in results if data is not None}

async def fetch_batch_history_data(stock_codes, days=30, session=None):
    """
    异步批量获取股票历史数据
    
    参数：
        stock_codes (list): 股票代码列表
        days (int): 需要获取的天数
        session (aiohttp.ClientSession, optional): 复用的会话对象
        
    返回：
        dict: 股票代码到DataFrame的映射，包含历史数据
        
    示例：
        >>> async with aiohttp.ClientSession() as session:
        ...     data = await fetch_batch_history_data(['000001', '600000'], days=30, session=session)
    """
    try:
        # 如果没有传入session，创建新的session
        if session is None:
            async with aiohttp.ClientSession() as session:
                return await _fetch_batch_history_data(stock_codes, days, session)
        else:
            return await _fetch_batch_history_data(stock_codes, days, session)
    except Exception as e:
        st.error(f"批量获取历史数据时出错: {str(e)}")
        return {}

async def _fetch_batch_history_data(stock_codes, days, session):
    """实际的批量数据获取逻辑"""
    try:
        # 设置日期范围
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=days*2)
        
        # 构建批量请求参数
        codes = []
        for code in stock_codes:
            if code.startswith('6'):
                codes.append(f"1.{code}")  # 上证
            elif code.startswith('8'):
                codes.append(f"2.{code}")  # 北交所
            else:
                codes.append(f"0.{code}")  # 深证
                
        codes_str = ",".join(codes)
        
        # 使用东方财富批量接口
        url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",  # 日线
            "fqt": "1",    # 前复权
            "beg": start_date.strftime('%Y%m%d'),
            "end": end_date.strftime('%Y%m%d'),
            "secids": codes_str,
            "ut": "fa5fd1943c7b386f172d6893dbfba244"
        }
        
        async with session.get(url, params=params) as response:
            data = await response.json()
            
            result = {}
            if data and "data" in data:
                for stock_data in data["data"]:
                    code = stock_data["code"]
                    klines = stock_data.get("klines", [])
                    
                    # 转换数据格式
                    records = []
                    for line in klines:
                        fields = line.split(',')
                        records.append({
                            '日期': fields[0],
                            '开盘': float(fields[1]),
                            '收盘': float(fields[2]),
                            '最高': float(fields[3]),
                            '最低': float(fields[4]),
                            '成交量': float(fields[5]),
                            '成交额': float(fields[6]),
                            '振幅': float(fields[7]),
                            '涨跌幅': float(fields[8]),
                            '涨跌额': float(fields[9]),
                            '换手率': float(fields[10])
                        })
                    
                    df = pd.DataFrame(records)
                    df['日期'] = pd.to_datetime(df['日期'])
                    df = df.sort_values('日期').tail(days)
                    
                    # 保存到缓存
                    stock_cache.save_to_cache(code, df)
                    result[code] = df
                    
            return result
            
    except Exception as e:
        st.error(f"获取历史数据时出错: {str(e)}")
        return {}

async def get_stock_data_async(stock_codes, days=30):
    """
    异步获取多只股票的历史数据
    
    首先检查本地缓存，对于未缓存的数据进行批量请求。
    
    参数：
        stock_codes (list): 股票代码列表
        days (int): 需要获取的天数
        
    返回：
        dict: 股票代码到DataFrame的映射
        
    示例：
        >>> data_dict = asyncio.run(get_stock_data_async(['000001', '600000']))
    """
    try:
        # 检查缓存
        result = {}
        codes_to_fetch = []
        
        for code in stock_codes:
            cached_data = stock_cache.get_cached_data(code)
            if cached_data is not None:
                result[code] = cached_data.tail(days)
            else:
                codes_to_fetch.append(code)
        
        if codes_to_fetch:
            # 批量获取未缓存的数据
            async with aiohttp.ClientSession() as session:
                batch_size = 50  # 每批处理的股票数量
                for i in range(0, len(codes_to_fetch), batch_size):
                    batch_codes = codes_to_fetch[i:i + batch_size]
                    batch_data = await fetch_batch_history_data(batch_codes, days, session)
                    result.update(batch_data)
                    
                    # 添加短暂延时避免请求过快
                    await asyncio.sleep(0.5)
        
        return result
        
    except Exception as e:
        st.error(f"异步获取股票数据时出错: {str(e)}")
        return {}

def get_stock_data(stock_code, days=30):
    """获取单只股票数据（兼容旧接口）"""
    try:
        # 使用异步方式获取数据
        result = asyncio.run(get_stock_data_async([stock_code], days))
        return result.get(stock_code)
        
    except Exception as e:
        st.error(f"获取股票 {stock_code} 数据时发生错误: {str(e)}")
        import traceback
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None