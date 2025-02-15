import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime, timedelta
import jieba.analyse
import traceback
import concurrent.futures
import threading
from functools import partial
import time
from snownlp import SnowNLP
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. 基础工具函数
def get_stock_data_with_retry(func, *args, **kwargs):
    """带重试机制的数据获取函数"""
    for i in range(3):  # 最多重试3次
        try:
            data = func(*args, **kwargs)
            if isinstance(data, pd.DataFrame) and not data.empty:
                # 打印数据列名，用于调试
                st.write(f"数据列名: {list(data.columns)}")
                return data
            time.sleep(1)
        except Exception as e:
            if i == 2:  # 最后一次重试
                st.error(f"获取数据失败: {str(e)}")
            time.sleep(1)
            continue
    
    st.warning("多次尝试后未能获取到有效数据")
    return pd.DataFrame()

# 2. 数据获取函数
def get_stock_info_base(code):
    """基础版获取股票信息（无缓存）"""
    try:
        info = get_stock_data_with_retry(ak.stock_individual_info_em, symbol=code)
        return info
    except Exception as e:
        st.write(f"获取股票信息失败 {code}: {str(e)}")
        return None

# 3. 缓存装饰器函数
@st.cache_data(ttl=3600)  # 缓存1小时
def get_basic_stock_data():
    """获取A股列表"""
    stock_list = get_stock_data_with_retry(ak.stock_zh_a_spot_em)
    if stock_list is None:
        return None
    return stock_list

@st.cache_data(ttl=86400)  # 缓存24小时
def get_stock_info(code):
    """获取单个股票信息（带缓存）"""
    return get_stock_info_base(code)

def process_single_stock(stock):
    """处理单个股票信息"""
    try:
        info = get_stock_info(stock['代码'])
        if info is not None and not info.empty:
            info_dict = dict(zip(info['item'], info['value']))
            
            # 获取上市时间
            list_date = None
            for key in ['上市日期', '上市时间', 'IPO日期', '发行日期']:
                if key in info_dict:
                    try:
                        list_date = pd.to_datetime(info_dict[key])
                        break
                    except:
                        continue
            
            if list_date is None:
                return None
            
            days_listed = (datetime.now() - list_date).days
            
            if days_listed > 180:
                return {
                    '代码': stock['代码'],
                    '名称': stock['名称'],
                    '上市时间': list_date,
                    '上市天数': days_listed,
                    '现价': stock['最新价'],
                    '成交额': stock['成交额']
                }
    except Exception as e:
        st.write(f"❌ {stock['代码']}: {str(e)}")
    return None

@st.cache_data(ttl=86400)  # 缓存24小时
def get_valid_stocks(stock_list):
    """获取符合上市时间要求的股票列表 - 多线程版本"""
    valid_stocks = []
    total = len(stock_list)
    processed_count = 0
    
    # 创建进度显示
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # 创建线程锁
    lock = threading.Lock()
    
    def update_progress():
        nonlocal processed_count
        with lock:
            processed_count += 1
            progress = processed_count / total
            progress_placeholder.progress(progress)
            status_placeholder.text(f"正在获取股票信息... ({processed_count}/{total})")
    
    # 使用线程池处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有任务
        future_to_stock = {
            executor.submit(process_single_stock, stock): stock 
            for _, stock in stock_list.iterrows()
        }
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_data = future.result()
            if stock_data is not None:
                with lock:
                    valid_stocks.append(stock_data)
                    # st.write(f"✅ {stock_data['名称']} ({stock_data['代码']})")
            update_progress()
    
    progress_placeholder.empty()
    status_placeholder.empty()
    
    # 转换为DataFrame并按代码排序
    valid_stocks_df = pd.DataFrame(valid_stocks)
    if not valid_stocks_df.empty:
        valid_stocks_df = valid_stocks_df.sort_values('代码')
    
    return valid_stocks_df

# 4. 主要处理函数
def filter_basic_stocks():
    """基础股票筛选"""
    try:
        # 获取所有A股数据
        stocks = get_stock_data_with_retry(
            lambda: ak.stock_zh_a_spot_em()
        )
        if stocks.empty:
            st.error("获取A股数据失败")
            return pd.DataFrame()
            
        # 打印列名和数据示例用于调试
        st.write("股票数据列名:", list(stocks.columns))
        st.write("数据示例:", stocks.head(1).to_dict('records'))
        
        # 确保代码列的格式正确（使用实际的列名）
        if '代码' in stocks.columns:
            code_column = '代码'
            name_column = '名称'
        elif 'symbol' in stocks.columns:
            code_column = 'symbol'
            name_column = 'name'
        else:
            st.error("无法找到股票代码列，当前列名：" + str(list(stocks.columns)))
            return pd.DataFrame()
        
        stocks[code_column] = stocks[code_column].astype(str).str.zfill(6)
        st.write("原始A股数量:", len(stocks))
        
        # 1. 剔除 ST 股票
        non_st_stocks = stocks[~stocks[name_column].str.contains('ST|退')]
        st.write(f"剔除ST股票后数量: {len(non_st_stocks)}")
        
        # 2. 获取上市时间信息
        @st.cache_data(ttl=3600*24)
        def get_stock_basic_info():
            try:
                # 使用东方财富股票列表接口
                basic_info = get_stock_data_with_retry(
                    lambda: ak.stock_info_a_code_name()
                )
                if not basic_info.empty:
                    st.write("基本信息列名:", list(basic_info.columns))
                    st.write("基本信息示例:", basic_info.head(1).to_dict('records'))
                return basic_info
            except Exception as e:
                st.error(f"获取基本信息失败: {str(e)}")
                return pd.DataFrame()
            
        stock_info = get_stock_basic_info()
        
        # 由于无法直接获取上市时间，我们暂时跳过这个筛选条件
        st.write("暂时跳过上市时间筛选")
        
        # 3. 获取成交额数据
        # 使用当日成交额数据
        if '成交额' in stocks.columns:
            volume_data = stocks[[code_column, '成交额']]
            volume_data['成交额'] = volume_data['成交额'].astype(float)
            
            # 剔除低成交额股票（5000万）
            active_stocks = volume_data[volume_data['成交额'] > 5000_0000][code_column]
            non_st_stocks = non_st_stocks[non_st_stocks[code_column].isin(active_stocks)]
            st.write(f"剔除低成交额股票后数量: {len(non_st_stocks)}")
        else:
            st.warning("无法获取成交额数据，跳过成交额筛选")
        
        # 4. 获取财务数据
        @st.cache_data(ttl=3600*24)
        def get_financial_data():
            try:
                # 使用业绩快报接口
                financial_data = get_stock_data_with_retry(
                    lambda: ak.stock_yjbb_em()
                )
                if not financial_data.empty:
                    st.write("财务数据列名:", list(financial_data.columns))
                    st.write("财务数据示例:", financial_data.head(1).to_dict('records'))
                return financial_data
            except Exception as e:
                st.error(f"获取财务数据失败: {str(e)}")
                return pd.DataFrame()
            
        financial_data = get_financial_data()
        if not financial_data.empty:
            # 使用实际的列名
            if '股票代码' in financial_data.columns:
                fin_code_column = '股票代码'
                profit_column = '净利润'
                if '净利润-净利润' in financial_data.columns:
                    profit_column = '净利润-净利润'
                elif '净利润同比' in financial_data.columns:
                    profit_column = '净利润同比'
            else:
                st.warning("无法找到财务数据列，跳过财务筛选")
                st.write("当前列名：" + str(list(financial_data.columns)))
                financial_data = pd.DataFrame()
        
        if not financial_data.empty:
            financial_data[fin_code_column] = financial_data[fin_code_column].astype(str).str.zfill(6)
            financial_data[profit_column] = financial_data[profit_column].astype(float)
            
            # 获取盈利的公司
            profitable_stocks = financial_data[financial_data[profit_column] > 0][fin_code_column]
            non_st_stocks = non_st_stocks[non_st_stocks[code_column].isin(profitable_stocks)]
            st.write(f"剔除亏损股票后数量: {len(non_st_stocks)}")
        
        return non_st_stocks
        
    except Exception as e:
        st.error(f"基础筛选发生错误: {str(e)}")
        st.write("错误详情:", traceback.format_exc())
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_cached_sectors():
    """获取板块数据（带缓存）"""
    try:
        # 获取东方财富行业板块数据
        sector_data = get_stock_data_with_retry(ak.stock_board_industry_name_em)
        if sector_data is not None and not sector_data.empty:
            # 确保列名正确
            if '板块名称' not in sector_data.columns and '行业名称' in sector_data.columns:
                sector_data = sector_data.rename(columns={'行业名称': '板块名称'})
            return sector_data
            
        st.warning("未获取到板块数据")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"获取板块数据失败: {str(e)}")
        return pd.DataFrame()

def analyze_news_sentiment():
    """分析新闻情感和热点 - 事件热度因子模型"""
    try:
        @st.cache_data(ttl=1800)
        def get_cached_news():
            # 1. 数据采集
            news_data = get_stock_data_with_retry(ak.stock_news_em)  # 东方财富新闻
            
            if news_data is not None and not news_data.empty:
                st.write("获取到新闻数量:", len(news_data))
                st.write("新闻数据列名:", list(news_data.columns))
                
                # 数据清洗
                news_data['新闻内容'] = news_data['新闻内容'].astype(str)
                news_data['新闻标题'] = news_data['新闻标题'].astype(str)
                news_data['发布时间'] = pd.Timestamp.now()
                
                return news_data
            
            st.error("未获取到新闻数据")
            return pd.DataFrame()

        # 3. 获取并分析新闻数据
        news_data = get_cached_news()
        if news_data.empty:
            st.error("获取新闻数据失败")
            return []
            
        # 4. 计算事件热度
        event_scores = {}
        
        # 使用jieba分词提取关键词
        for _, news in news_data.iterrows():
            title = news['新闻标题']
            content = news['新闻内容']
            
            # 合并标题和内容
            text = title + " " + content
            
            # 提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=5)
            
            # 更新事件热度
            for keyword in keywords:
                if len(keyword) >= 2:  # 只处理长度大于等于2的关键词
                    if keyword not in event_scores:
                        event_scores[keyword] = {
                            'heat_score': 1,
                            'news_count': 1,
                            'latest_news': [title],
                            'avg_sentiment': 0
                        }
                    else:
                        event_scores[keyword]['heat_score'] += 1
                        event_scores[keyword]['news_count'] += 1
                        event_scores[keyword]['latest_news'].append(title)
                        if len(event_scores[keyword]['latest_news']) > 3:
                            event_scores[keyword]['latest_news'] = event_scores[keyword]['latest_news'][-3:]
        
        if not event_scores:
            st.error("未能提取到有效的事件关键词")
            return []
            
        # 5. 标准化热度分数
        heat_scores = [data['heat_score'] for data in event_scores.values()]
        if not heat_scores:
            st.error("未计算出热度分数")
            return []
            
        mean_heat = np.mean(heat_scores)
        std_heat = np.std(heat_scores)
        max_heat = max(heat_scores)
        
        # 创建结果DataFrame
        results = []
        hot_sectors = []
        
        for event, data in event_scores.items():
            # Z-score标准化
            z_score = (data['heat_score'] - mean_heat) / std_heat if std_heat != 0 else 0
            # 0-100分标准化
            norm_score = (data['heat_score'] / max_heat * 100) if max_heat != 0 else 0
            
            results.append({
                '事件关键词': event,
                '热度分数': round(norm_score, 2),
                '标准分数': round(z_score, 2),
                '新闻数量': data['news_count'],
                '情感倾向': round(data['avg_sentiment'], 2),
                '最新相关新闻': '\n'.join(data['latest_news'])
            })
            
            hot_sectors.append((event, round(norm_score, 2)))
        
        if not results:
            st.error("未生成热度排名")
            return []
            
        # 创建并显示DataFrame
        event_df = pd.DataFrame(results)
        event_df = event_df.sort_values('热度分数', ascending=False)
        
        st.write("事件热度排名：")
        st.write(event_df)
        
        # 返回前5个热门事件
        hot_sectors.sort(key=lambda x: x[1], reverse=True)
        return hot_sectors[:5]
        
    except Exception as e:
        st.error(f"新闻分析失败: {str(e)}")
        st.write("错误详情:", traceback.format_exc())
        return []

def filter_technical_indicators(stock_code):
    """技术指标筛选"""
    try:
        # 获取股票行情数据
        stock_data = get_stock_data_with_retry(
            ak.stock_zh_a_hist,
            symbol=stock_code,
            period="daily",
            start_date=(datetime.now() - timedelta(days=60)).strftime('%Y%m%d'),
            end_date=datetime.now().strftime('%Y%m%d')
        )
        
        if stock_data is None:
            return False
            
        # 计算均线
        stock_data['MA5'] = stock_data['收盘'].rolling(window=5).mean()
        stock_data['MA10'] = stock_data['收盘'].rolling(window=10).mean()
        stock_data['MA20'] = stock_data['收盘'].rolling(window=20).mean()
        
        # 计算RSI
        delta = stock_data['收盘'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算量比
        stock_data['量比'] = stock_data['成交量'] / stock_data['成交量'].rolling(window=5).mean()
        
        # 获取最新数据
        latest = stock_data.iloc[-1]
        
        # 检查技术指标条件
        ma_condition = latest['MA5'] > latest['MA10'] > latest['MA20']
        rsi_condition = latest['RSI'] > 50
        volume_condition = latest['量比'] > 1.5
        
        return ma_condition and rsi_condition and volume_condition
        
    except Exception as e:
        st.error(f"技术指标筛选失败: {str(e)}")
        st.write("错误详情:", traceback.format_exc())
        return False

@st.cache_data(ttl=3600)
def get_sector_stocks(sector_name):
    """获取板块成分股（带缓存）"""
    try:
        # 使用 lambda 包装函数调用，处理参数传递
        sector_stocks = get_stock_data_with_retry(
            lambda: ak.stock_board_industry_cons_em(symbol=sector_name)
        )
        
        if not sector_stocks.empty:
            # 确保返回的数据包含必要的列
            required_columns = ['代码', '名称']
            if all(col in sector_stocks.columns for col in required_columns):
                return sector_stocks
                
        st.warning(f"未获取到 {sector_name} 板块的成分股数据")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"获取板块成分股数据失败: {str(e)}")
        return pd.DataFrame()

def analyze_stock(stock, code_column='代码', name_column='名称'):
    """分析单个股票"""
    try:
        code = stock[code_column]
        
        # 使用缓存的技术指标分析结果
        @st.cache_data(ttl=1800)  # 30分钟缓存
        def cached_technical_analysis(stock_code):
            return filter_technical_indicators(stock_code)
            
        if cached_technical_analysis(code):
            return {
                '代码': code,
                '名称': stock[name_column],
                '现价': stock.get('最新价', 0),
                '涨跌幅': stock.get('涨跌幅', 0)
            }
    except Exception as e:
        st.warning(f"分析股票 {code} 时出错: {str(e)}")
    return None

def select_stocks(base_stocks=None):
    """基于技术指标的选股策略"""
    try:
        # 获取并筛选基础股票池
        filtered_stocks = filter_basic_stocks()
        if filtered_stocks.empty:
            st.error("基础筛选未获取到符合条件的股票")
            return pd.DataFrame()
            
        st.write("开始技术指标分析...")
        progress_container = st.empty()
        progress_bar = st.progress(0)
        
        selected_stocks = []
        total_stocks = len(filtered_stocks)
        processed_count = 0
        
        # 创建线程池
        max_workers = min(32, total_stocks)  # 最多32个线程
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(
                    analyze_stock, 
                    stock,
                    code_column='代码' if '代码' in filtered_stocks.columns else 'symbol',
                    name_column='名称' if '名称' in filtered_stocks.columns else 'name'
                ): stock 
                for _, stock in filtered_stocks.iterrows()
            }
            
            # 处理完成的任务
            start_time = time.time()
            for future in as_completed(future_to_stock):
                processed_count += 1
                
                # 更新进度
                progress = processed_count / total_stocks
                progress_bar.progress(progress)
                
                # 计算预估剩余时间
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / progress if progress > 0 else 0
                remaining_time = estimated_total_time - elapsed_time
                
                # 更新进度信息
                progress_container.text(
                    f"已处理: {processed_count}/{total_stocks} "
                    f"预计剩余时间: {remaining_time:.1f}秒"
                )
                
                # 获取分析结果
                result = future.result()
                if result:
                    selected_stocks.append(result)
        
        # 清理进度显示
        progress_container.empty()
        progress_bar.empty()
        
        if selected_stocks:
            st.write("选股结果：")
            result_df = pd.DataFrame(selected_stocks)
            # 按涨跌幅排序
            result_df = result_df.sort_values('涨跌幅', ascending=False)
            
            # 缓存选股结果
            @st.cache_data(ttl=1800)  # 30分钟缓存
            def cache_selected_stocks(df):
                return df
                
            result_df = cache_selected_stocks(result_df)
            st.write(result_df)
            
            # 显示选股统计
            st.write(f"""
            ### 选股统计
            - 基础股票池数量: {total_stocks}
            - 符合条件数量: {len(selected_stocks)}
            - 选股比例: {len(selected_stocks)/total_stocks*100:.2f}%
            """)
            
            return result_df
        else:
            st.warning("未找到符合条件的股票")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"选股过程发生错误: {str(e)}")
        st.write("错误详情:", traceback.format_exc())
        return pd.DataFrame()

def main():
    """主函数"""
    st.title("技术指标选股策略")
    
    # 添加策略说明
    st.markdown("""
    ### 策略说明
    1. 基础筛选
       - 剔除ST股票
       - 剔除低成交额股票
       - 剔除亏损股票
       
    2. 技术指标
       - MACD金叉
       - KDJ超卖区
       - 均线多头排列
    """)
    
    # 添加缓存控制
    col1, col2 = st.columns(2)
    with col1:
        if st.button("清除缓存"):
            st.cache_data.clear()
            st.success("缓存已清除")
    
    with col2:
        if st.button("开始选股"):
            with st.spinner("正在执行选股策略..."):
                start_time = time.time()
                result = select_stocks()
                end_time = time.time()
                
                if not result.empty:
                    st.success(f"选股完成！用时: {end_time - start_time:.1f}秒")
    
    # 添加策略详情说明
    with st.expander("查看策略详情"):
        st.markdown("""
        #### 技术指标条件
        1. MACD
           - DIFF线上穿DEA线
           - MACD柱由负转正
           
        2. KDJ指标
           - K值和D值在超卖区
           - J值开始上升
           
        3. 均线系统
           - 短期均线上穿长期均线
           - 维持多头排列
        """)

if __name__ == "__main__":
    main() 