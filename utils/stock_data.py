import akshare as ak
import pandas as pd
from .technical_analysis import check_buy_signal
import time
import streamlit as st
import concurrent.futures
import threading
from queue import Queue
from requests.exceptions import RequestException, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError
import random
from functools import wraps

# 创建一个线程安全的队列来存储进度信息
progress_queue = Queue()
selected_stocks_queue = Queue()

# 自定义重试装饰器
def retry(stop_max_attempt_number=3, 
         wait_fixed=2000, 
         wait_exponential_multiplier=None, 
         wait_exponential_max=None,
         retry_on_exception=None):
    """
    自定义重试装饰器
    :param stop_max_attempt_number: 最大重试次数
    :param wait_fixed: 固定等待时间（毫秒）
    :param wait_exponential_multiplier: 指数退避乘数（毫秒）
    :param wait_exponential_max: 最大等待时间（毫秒）
    :param retry_on_exception: 需要重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_attempts = stop_max_attempt_number
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 检查是否是需要重试的异常类型
                    if retry_on_exception and not retry_on_exception(e):
                        raise e
                        
                    if attempt == max_attempts - 1:  # 最后一次尝试
                        raise e
                        
                    # 计算等待时间
                    if wait_exponential_multiplier:
                        wait_time = min(
                            wait_exponential_multiplier * (2 ** attempt),
                            wait_exponential_max or float('inf')
                        ) / 1000  # 转换为秒
                    else:
                        wait_time = wait_fixed / 1000
                        
                    time.sleep(wait_time)
                    
            return None
        return wrapper
    return decorator

def should_retry(exception):
    """判断是否需要重试的条件"""
    retry_exceptions = (
        ConnectionError,
        TimeoutError,
        Exception  # 临时添加，用于处理所有异常
    )
    return isinstance(exception, retry_exceptions)

@retry(
    stop_max_attempt_number=5,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    retry_on_exception=should_retry
)
def get_stock_data(stock_code, days=30):
    """获取股票历史数据"""
    try:
        stock_code = stock_code.strip()
        # 随机延时0.5-2秒
        time.sleep(random.uniform(0.5, 2))
        
        df = ak.stock_zh_a_hist(
            symbol=stock_code, 
            period="daily", 
            adjust="qfq",
            timeout=30
        )
        
        if df is None or df.empty:
            st.warning(f"无法获取股票 {stock_code} 的数据")
            return None
            
        return df.tail(days)
        
    except Exception as e:
        st.error(f"获取股票 {stock_code} 数据时发生错误: {str(e)}")
        raise

@retry(
    stop_max_attempt_number=5,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    retry_on_exception=should_retry
)
def get_stock_list_with_retry():
    """获取股票列表（带重试机制）"""
    try:
        # 随机延时0.5-2秒
        time.sleep(random.uniform(0.5, 2))
        return ak.stock_info_a_code_name()
    except Exception as e:
        print(f"获取股票列表时发生错误: {str(e)}")
        raise

@retry(
    stop_max_attempt_number=5,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    retry_on_exception=should_retry
)
def get_realtime_quotes_with_retry():
    """获取实时行情（带重试机制）"""
    try:
        # 随机延时0.5-2秒
        time.sleep(random.uniform(0.5, 2))
        return ak.stock_zh_a_spot_em()
    except Exception as e:
        print(f"获取实时行情时发生错误: {str(e)}")
        raise

def process_stock(stock, latest_quotes):
    """处理单个股票的函数"""
    try:
        # 获取历史数据
        stock_data = get_stock_data(stock['代码'])
        
        if stock_data is not None and not stock_data.empty:
            # 检查买入信号
            is_buy, reason = check_buy_signal(stock_data)
            
            if is_buy:
                # 获取该股票的最新行情
                stock_quote = latest_quotes[latest_quotes['代码'] == stock['代码']]
                
                if not stock_quote.empty:
                    selected_stock = {
                        '代码': stock['代码'],
                        '名称': stock['名称'],
                        '现价': float(stock_quote['现价'].iloc[0]),
                        '涨跌幅': float(stock_quote['涨跌幅'].iloc[0]),
                        '换手率': float(stock_quote['换手率'].iloc[0]),
                        '成交额': float(stock_quote['成交额'].iloc[0]) / 100,
                        '选股理由': reason
                    }
                    selected_stocks_queue.put(selected_stock)
                    return True
        
        return False
    except Exception as e:
        st.error(f"处理股票 {stock['代码']} 时发生错误: {str(e)}")
        return False

def update_progress():
    """更新进度条的函数"""
    progress_bar = st.progress(0.0)
    while True:
        progress = progress_queue.get()
        if progress == -1:  # 结束信号
            progress_bar.progress(1.0)
            break
        progress_bar.progress(progress)

def get_stock_list():
    """获取A股股票列表并进行策略筛选"""
    @retry(
        stop_max_attempt_number=5,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000,
        retry_on_exception=should_retry
    )
    def _get_stock_info():
        time.sleep(random.uniform(0.5, 2))
        return ak.stock_info_a_code_name()
        
    @retry(
        stop_max_attempt_number=5,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000,
        retry_on_exception=should_retry
    )
    def _get_realtime_quotes():
        time.sleep(random.uniform(0.5, 2))
        return ak.stock_zh_a_spot_em()
    
    try:
        # 获取所有A股信息（带重试）
        stock_info = _get_stock_info()
        stock_info.columns = ['代码', '名称']
        st.write(f"获取到总股票数量: {len(stock_info)}")
        
        # 排除ST股票
        non_st_stocks = stock_info[~stock_info['名称'].str.contains('ST|退')]
        st.write(f"排除ST后股票数量: {len(non_st_stocks)}")
        
        # 获取实时行情数据（带重试）
        latest_quotes = _get_realtime_quotes()
        latest_quotes = latest_quotes.rename(columns={
            '代码': '代码',
            '名称': '名称',
            '最新价': '现价',
            '涨跌幅': '涨跌幅',
            '换手率': '换手率',
            '成交额': '成交额'
        })
        st.write(f"获取到实时行情数量: {len(latest_quotes)}")
        st.dataframe(latest_quotes)
        # 清空队列
        while not progress_queue.empty():
            progress_queue.get()
        while not selected_stocks_queue.empty():
            selected_stocks_queue.get()
        
        # 启动进度更新线程
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()
        
        # 使用线程池处理股票
        total_stocks = len(non_st_stocks)
        processed_count = 0
        selected_stocks = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有股票处理任务
            future_to_stock = {
                executor.submit(process_stock, stock, latest_quotes): stock 
                for _, stock in non_st_stocks.iterrows()
            }
            
            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_stock):
                processed_count += 1
                progress = min(0.99, processed_count / total_stocks)
                progress_queue.put(progress)
                
                # 检查是否有新的选中股票
                while not selected_stocks_queue.empty():
                    selected_stocks.append(selected_stocks_queue.get())
        
        # 发送结束信号给进度更新线程
        progress_queue.put(-1)
        progress_thread.join()
        
        if not selected_stocks:
            st.warning("未找到符合条件的股票")
            return pd.DataFrame()
        
        # 创建DataFrame并进行数据类型转换
        result_df = pd.DataFrame(selected_stocks)
        
        # 确保数值列的类型正确
        numeric_columns = ['现价', '涨跌幅', '换手率', '成交额']
        for col in numeric_columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        
        # 对结果进行排序
        result_df = result_df.sort_values('涨跌幅', ascending=False)
        
        st.write(f"分析完成，共找到 {len(result_df)} 只符合条件的股票")
        
        return result_df
        
    except Exception as e:
        st.error(f"选股过程发生错误: {str(e)}")
        return pd.DataFrame()
    finally:
        # 确保进度更新线程结束
        if 'progress_queue' in locals():
            progress_queue.put(-1) 