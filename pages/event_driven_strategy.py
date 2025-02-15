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

def get_batch_market_data():
    """批量获取市场数据"""
    try:
        data = {}
        
        # 1. 获取市场行情数据（包含换手率和涨跌幅）
        @st.cache_data(ttl=1800)
        def get_market_data():
            try:
                market_data = get_stock_data_with_retry(
                    lambda: ak.stock_zh_a_spot_em()
                )
                
                if market_data.empty:
                    st.warning("获取市场数据为空")
                    return pd.DataFrame()
                    
                st.write("市场数据列名:", list(market_data.columns))
                return market_data
                
            except Exception as e:
                st.warning(f"获取市场数据失败: {str(e)}")
                return pd.DataFrame()
        
        market_data = get_market_data()
        if not market_data.empty:
            data['market'] = market_data
            
            # 提取行业信息
            if '所属行业' in market_data.columns:
                industry_dict = {}
                for _, row in market_data.iterrows():
                    industry = row['所属行业']
                    code = row['代码']
                    if industry not in industry_dict:
                        industry_dict[industry] = []
                    industry_dict[industry].append(code)
                data['industry'] = industry_dict
            
            # 提取换手率数据
            if '换手率' in market_data.columns:
                data['turnover'] = market_data[['代码', '换手率']]
        
        # 2. 获取龙虎榜数据（可选）
        @st.cache_data(ttl=1800)
        def get_top_list_summary():
            try:
                # 使用东方财富龙虎榜单接口
                top_list = get_stock_data_with_retry(
                    lambda: ak.stock_lhb_detail_em()
                )
                
                if top_list.empty:
                    st.warning("获取龙虎榜数据为空")
                    return pd.DataFrame()
                    
                st.write("龙虎榜数据列名:", list(top_list.columns))
                
                # 如果需要，可以进行数据清洗和转换
                if '代码' not in top_list.columns and '股票代码' in top_list.columns:
                    top_list = top_list.rename(columns={'股票代码': '代码'})
                
                return top_list
                
            except Exception as e:
                st.warning(f"获取龙虎榜数据失败: {str(e)}")
                return pd.DataFrame()
            
        data['top_list'] = get_top_list_summary()
        
        # 打印获取到的数据状态
        st.write("数据获取状态:")
        st.write(f"- 市场数据: {'已获取' if 'market' in data else '未获取'}")
        if 'market' in data:
            st.write(f"  - 数据行数: {len(data['market'])}")
            st.write(f"  - 数据列: {list(data['market'].columns)}")
        st.write(f"- 行业数据: {len(data.get('industry', {})) if 'industry' in data else 0} 个行业")
        st.write(f"- 换手率数据: {'已获取' if 'turnover' in data else '未获取'}")
        st.write(f"- 龙虎榜数据: {'已获取' if not data.get('top_list', pd.DataFrame()).empty else '未获取'}")
        
        # 数据示例
        for key, value in data.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                st.write(f"\n{key} 数据示例:")
                st.write(value.head(1))
                st.write(f"数据形状: {value.shape}")
            elif isinstance(value, dict) and value:
                st.write(f"\n{key} 数据示例:")
                sample_key = next(iter(value))
                st.write(f"- {sample_key}: {value[sample_key][:5] if value[sample_key] else '空'}")
        
        return data
        
    except Exception as e:
        st.error(f"获取市场数据时出错: {str(e)}")
        st.write("错误详情:", traceback.format_exc())
        return {}

def analyze_market_strength(stock_code, stock_data, market_data):
    """分析市场强度"""
    try:
        if not market_data or 'market' not in market_data:
            return True
            
        market_df = market_data['market']
        
        # 1. 计算行业涨幅排名
        if '所属行业' in market_df.columns:
            stock_row = market_df[market_df['代码'] == stock_code]
            if not stock_row.empty:
                industry = stock_row['所属行业'].iloc[0]
                industry_stocks = market_df[market_df['所属行业'] == industry]
                
                if not industry_stocks.empty:
                    industry_stocks = industry_stocks.sort_values('涨跌幅', ascending=False)
                    total_stocks = len(industry_stocks)
                    current_rank = industry_stocks[industry_stocks['代码'] == stock_code].index[0]
                    rank_percent = current_rank / total_stocks
                    
                    if rank_percent > 0.3:  # 不在前30%
                        return False
        
        # 2. 分析换手率
        if '换手率' in market_df.columns:
            stock_row = market_df[market_df['代码'] == stock_code]
            if not stock_row.empty:
                current_turnover = float(stock_row['换手率'].iloc[0])
                if current_turnover < 3:  # 换手率过低
                    return False
                    
        return True
        
    except Exception as e:
        st.warning(f"分析市场强度时出错 {stock_code}: {str(e)}")
        return True

def analyze_capital_flow(stock_code, market_data):
    """分析资金流向"""
    try:
        if not market_data:
            return True
            
        # 分析龙虎榜
        top_list = market_data.get('top_list', pd.DataFrame())
        if not top_list.empty and '代码' in top_list.columns:
            stock_top = top_list[top_list['代码'] == stock_code]
            if not stock_top.empty:
                return True  # 上榜即视为资金关注
                    
        # 如果没有上榜，检查换手率变化
        market_df = market_data.get('market', pd.DataFrame())
        if not market_df.empty and '换手率' in market_df.columns:
            stock_row = market_df[market_df['代码'] == stock_code]
            if not stock_row.empty:
                turnover = float(stock_row['换手率'].iloc[0])
                if turnover > 5:  # 换手率大于5%视为资金活跃
                    return True
                    
        return False  # 如果以上条件都不满足，返回False
        
    except Exception as e:
        st.warning(f"分析资金流向时出错 {stock_code}: {str(e)}")
        return True

def analyze_stock(stock_row, stock_data, market_data):
    """分析单个股票"""
    try:
        # 获取股票代码
        code = stock_row['代码']
        
        # 使用缓存的技术指标分析结果
        @st.cache_data(ttl=1800)  # 30分钟缓存
        def cached_technical_analysis(stock_code):
            return filter_technical_indicators(stock_code)
            
        # 1. 技术指标分析
        if not cached_technical_analysis(code):
            return None
            
        # 2. 市场强度分析
        if not analyze_market_strength(code, stock_data, market_data):
            return None
            
        # 3. 资金流向分析
        if not analyze_capital_flow(code, market_data):
            return None
            
        return {
            '代码': code,
            '名称': stock_row['名称'],
            '现价': stock_row.get('最新价', 0),
            '涨跌幅': stock_row.get('涨跌幅', 0),
            '换手率': stock_row.get('换手率', 0)
        }
        
    except Exception as e:
        st.warning(f"分析股票时出错: {str(e)}")
        st.write("错误详情:", traceback.format_exc())
    return None

def plot_stock_chart(code):
    """绘制股票走势图"""
    try:
        # 获取历史数据
        df = get_stock_data_with_retry(
            lambda: ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        )
        
        if df.empty:
            return "获取数据失败"
            
        # 计算技术指标
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['MA60'] = df['收盘'].rolling(window=60).mean()
        
        # 使用plotly绘制图表
        fig = go.Figure()
        
        # 添加K线图
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['开盘'],
            high=df['最高'],
            low=df['最低'],
            close=df['收盘'],
            name='K线'
        ))
        
        # 添加均线
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='MA5', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='MA60', line=dict(color='purple')))
        
        # 添加成交量
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['成交量'],
            name='成交量',
            yaxis='y2'
        ))
        
        # 更新布局
        fig.update_layout(
            title=f'{code} 走势图',
            yaxis_title='价格',
            yaxis2=dict(
                title='成交量',
                overlaying='y',
                side='right'
            ),
            xaxis_title='日期',
            height=600,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        return f"绘制图表错误: {str(e)}"

def display_stock_details():
    """显示股票详细信息"""
    try:
        # 从session_state获取数据
        code = st.session_state.current_stock
        result_df = st.session_state.selected_stocks_df
        
        if not code or result_df.empty:
            return
            
        # 使用session_state中的容器
        with st.session_state.detail_container.container():
            stock_name = result_df[result_df['代码']==code]['名称'].iloc[0]
            st.write(f"### {stock_name}({code}) 详细信息")
            
            # 创建三个标签页
            tab1, tab2, tab3 = st.tabs(["走势图", "技术指标", "资金流向"])
            
            with tab1:
                # 缓存图表配置
                if 'chart_days' not in st.session_state:
                    st.session_state.chart_days = 30
                if 'chart_freq' not in st.session_state:
                    st.session_state.chart_freq = "daily"
                
                # 添加日期选择
                col1, col2 = st.columns(2)
                with col1:
                    days = st.selectbox(
                        "选择时间范围",
                        options=[30, 60, 90, 180, 365],
                        format_func=lambda x: f"近{x}天",
                        key=f"days_selector_{code}",
                        index=0
                    )
                with col2:
                    freq = st.selectbox(
                        "选择周期",
                        options=["daily", "weekly", "monthly"],
                        format_func=lambda x: {"daily": "日K", "weekly": "周K", "monthly": "月K"}[x],
                        key=f"freq_selector_{code}",
                        index=0
                    )
                
                with st.spinner("正在加载走势图..."):
                    chart = plot_stock_chart(code)
                    if isinstance(chart, go.Figure):
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.error(chart)
            
            with tab2:
                # 显示技术指标详情
                st.write("### 技术指标")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("MACD指标")
                    st.write("- DIF: xxx")
                    st.write("- DEA: xxx")
                    st.write("- MACD: xxx")
                with col2:
                    st.write("KDJ指标")
                    st.write("- K: xxx")
                    st.write("- D: xxx")
                    st.write("- J: xxx")
                
            with tab3:
                # 显示资金流向
                st.write("### 资金流向")
                st.write("- 主力净流入: xxx万")
                st.write("- 累计换手率: xxx%")
                st.write("- 近期龙虎榜: xxx")
            
    except Exception as e:
        st.error(f"显示股票详情时出错: {str(e)}")
        st.write("错误详情:", traceback.format_exc())

def select_stocks(base_stocks=None):
    """基于技术指标的选股策略"""
    try:
        # 初始化 session_state
        if 'selected_stocks_df' not in st.session_state:
            st.session_state.selected_stocks_df = pd.DataFrame()
            
        # 只在需要时执行选股逻辑
        if st.session_state.selected_stocks_df.empty:
            # 获取并筛选基础股票池
            filtered_stocks = filter_basic_stocks()
            if filtered_stocks.empty:
                st.error("基础筛选未获取到符合条件的股票")
                return pd.DataFrame()
            
            # 批量获取市场数据
            st.write("正在获取市场数据...")
            market_data = get_batch_market_data()
            
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
                        filtered_stocks,
                        market_data
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
                result_df = pd.DataFrame(selected_stocks)
                result_df = result_df.sort_values('涨跌幅', ascending=False)
                st.session_state.selected_stocks_df = result_df
            else:
                st.warning("未找到符合条件的股票")
                return pd.DataFrame()
        
        # 使用缓存的结果
        result_df = st.session_state.selected_stocks_df
        
        # 显示结果表格
        st.write("### 选股结果")
        st.dataframe(
            result_df,
            column_config={
                "代码": st.column_config.TextColumn("代码"),
                "名称": st.column_config.TextColumn("名称"),
                "现价": st.column_config.NumberColumn("现价", format="%.2f"),
                "涨跌幅": st.column_config.NumberColumn("涨跌幅", format="%.2f%%"),
                "换手率": st.column_config.NumberColumn("换手率", format="%.2f%%"),
            },
            hide_index=True
        )
        
        # 创建一个空容器用于显示详细信息
        detail_container = st.empty()
        
        # 创建选择框
        selected_stock = st.selectbox(
            "选择要查看的股票",
            options=result_df['代码'].tolist(),
            format_func=lambda x: f"{x} - {result_df[result_df['代码']==x]['名称'].iloc[0]}"
        )
        
        # 如果选择了股票，显示详情
        if selected_stock:
            stock_name = result_df[result_df['代码']==selected_stock]['名称'].iloc[0]
            
            with detail_container.container():
                st.write(f"### {stock_name}({selected_stock}) 详细信息")
                
                # 创建三个标签页
                tab1, tab2, tab3 = st.tabs(["走势图", "技术指标", "资金流向"])
                
                with tab1:
                    # 添加日期选择
                    col1, col2 = st.columns(2)
                    with col1:
                        days = st.selectbox(
                            "选择时间范围",
                            options=[30, 60, 90, 180, 365],
                            format_func=lambda x: f"近{x}天",
                            key=f"days_selector_{selected_stock}"
                        )
                    with col2:
                        freq = st.selectbox(
                            "选择周期",
                            options=["daily", "weekly", "monthly"],
                            format_func=lambda x: {"daily": "日K", "weekly": "周K", "monthly": "月K"}[x],
                            key=f"freq_selector_{selected_stock}"
                        )
                    
                    with st.spinner("正在加载走势图..."):
                        chart = plot_stock_chart(selected_stock)
                        if isinstance(chart, go.Figure):
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.error(chart)
                
                with tab2:
                    # 使用缓存的技术指标数据
                    @st.cache_data(ttl=300)  # 5分钟缓存
                    def get_cached_technical_data(code):
                        return {
                            'macd': get_macd_data(code),
                            'kdj': get_kdj_data(code),
                            'ma': get_ma_data(code)
                        }
                    
                    tech_data = get_cached_technical_data(selected_stock)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("#### MACD指标")
                        if isinstance(tech_data['macd'], dict):
                            st.write(f"- DIF: {tech_data['macd']['dif']:.3f}")
                            st.write(f"- DEA: {tech_data['macd']['dea']:.3f}")
                            st.write(f"- MACD: {tech_data['macd']['macd']:.3f}")
                            st.write(f"- 信号: {tech_data['macd']['signal']}")
                        else:
                            st.error(tech_data['macd'])
                    
                    with col2:
                        st.write("#### KDJ指标")
                        if isinstance(tech_data['kdj'], dict):
                            st.write(f"- K: {tech_data['kdj']['k']:.2f}")
                            st.write(f"- D: {tech_data['kdj']['d']:.2f}")
                            st.write(f"- J: {tech_data['kdj']['j']:.2f}")
                            st.write(f"- 信号: {tech_data['kdj']['signal']}")
                        else:
                            st.error(tech_data['kdj'])
                    
                    with col3:
                        st.write("#### 均线系统")
                        if isinstance(tech_data['ma'], dict):
                            st.write(f"- MA5: {tech_data['ma']['ma5']:.2f}")
                            st.write(f"- MA20: {tech_data['ma']['ma20']:.2f}")
                            st.write(f"- MA60: {tech_data['ma']['ma60']:.2f}")
                            st.write(f"- 信号: {tech_data['ma']['signal']}")
                        else:
                            st.error(tech_data['ma'])
                
                with tab3:
                    # 使用缓存的资金流向数据
                    @st.cache_data(ttl=300)  # 5分钟缓存
                    def get_cached_flow_data(code):
                        return {
                            'flow': get_capital_flow(code),
                            'industry': get_industry_compare(code)
                        }
                    
                    flow_data = get_cached_flow_data(selected_stock)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("#### 资金流向")
                        if isinstance(flow_data['flow'], dict):
                            st.write(f"- 主力净流入: {flow_data['flow']['main_net_inflow']}万")
                            st.write(f"- 累计换手率: {flow_data['flow']['total_turnover']:.2f}%")
                            st.write(f"- 近期龙虎榜: {flow_data['flow']['top_list_status']}")
                        else:
                            st.error(flow_data['flow'])
                    
                    with col2:
                        st.write("#### 行业对比")
                        if isinstance(flow_data['industry'], dict):
                            st.write(f"- 所属行业: {flow_data['industry']['industry']}")
                            st.write(f"- 行业排名: {flow_data['industry']['rank']}/{flow_data['industry']['total']}")
                            st.write(f"- 相对强度: {flow_data['industry']['strength']}")
                            if 'performance' in flow_data['industry']:
                                st.write(f"- 涨跌幅: {flow_data['industry']['performance']}")
                        else:
                            st.error(flow_data['industry'])
        
        # 显示选股统计
        st.write(f"""
        ### 选股统计
        - 基础股票池数量: {len(result_df)}
        - 符合条件数量: {len(result_df)}
        - 选股比例: 100%
        """)
        
        return result_df
            
    except Exception as e:
        st.error(f"选股过程发生错误: {str(e)}")
        st.write("错误详情:", traceback.format_exc())
        return pd.DataFrame()

def get_macd_data(code):
    """获取MACD指标数据"""
    try:
        # 获取历史数据
        df = get_stock_data_with_retry(
            lambda: ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        )
        
        if df.empty:
            return "获取数据失败"
            
        # 计算MACD
        exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
        exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
        dif = exp1 - exp2
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = 2 * (dif - dea)
        
        # 判断信号
        if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2]:
            signal = "金叉买入"
        elif dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2]:
            signal = "死叉卖出"
        else:
            signal = "观望"
            
        return {
            'dif': dif.iloc[-1],
            'dea': dea.iloc[-1],
            'macd': macd.iloc[-1],
            'signal': signal
        }
        
    except Exception as e:
        return f"计算MACD失败: {str(e)}"

def get_kdj_data(code):
    """获取KDJ指标数据"""
    try:
        # 获取历史数据
        df = get_stock_data_with_retry(
            lambda: ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        )
        
        if df.empty:
            return "获取数据失败"
            
        # 计算KDJ
        low_list = df['最低'].rolling(window=9, min_periods=9).min()
        high_list = df['最高'].rolling(window=9, min_periods=9).max()
        rsv = (df['收盘'] - low_list) / (high_list - low_list) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d
        
        # 判断信号
        if k.iloc[-1] < 20 and j.iloc[-1] > j.iloc[-2]:
            signal = "超卖反转"
        elif k.iloc[-1] > 80 and j.iloc[-1] < j.iloc[-2]:
            signal = "超买反转"
        else:
            signal = "观望"
            
        return {
            'k': k.iloc[-1],
            'd': d.iloc[-1],
            'j': j.iloc[-1],
            'signal': signal
        }
        
    except Exception as e:
        return f"计算KDJ失败: {str(e)}"

def get_ma_data(code):
    """获取均线数据"""
    try:
        # 获取历史数据
        df = get_stock_data_with_retry(
            lambda: ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        )
        
        if df.empty:
            return "获取数据失败"
            
        # 计算均线
        ma5 = df['收盘'].rolling(window=5).mean()
        ma20 = df['收盘'].rolling(window=20).mean()
        ma60 = df['收盘'].rolling(window=60).mean()
        
        # 判断信号
        if ma5.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
            if ma5.iloc[-2] <= ma20.iloc[-2]:
                signal = "金叉买入"
            else:
                signal = "多头排列"
        elif ma5.iloc[-1] < ma20.iloc[-1] < ma60.iloc[-1]:
            if ma5.iloc[-2] >= ma20.iloc[-2]:
                signal = "死叉卖出"
            else:
                signal = "空头排列"
        else:
            signal = "观望"
            
        return {
            'ma5': ma5.iloc[-1],
            'ma20': ma20.iloc[-1],
            'ma60': ma60.iloc[-1],
            'signal': signal
        }
        
    except Exception as e:
        return f"计算均线失败: {str(e)}"

def get_capital_flow(code):
    """获取资金流向数据"""
    try:
        # 获取市场数据
        market_data = get_batch_market_data()
        
        if not market_data:
            return "获取数据失败"
            
        # 获取换手率
        turnover = 0
        market_df = market_data.get('market', pd.DataFrame())
        if not market_df.empty:
            stock_data = market_df[market_df['代码'] == code]
            if not stock_data.empty and '换手率' in stock_data.columns:
                turnover = stock_data['换手率'].iloc[0]
        
        # 获取龙虎榜状态
        top_list = market_data.get('top_list', pd.DataFrame())
        if not top_list.empty and '代码' in top_list.columns:
            stock_top = top_list[top_list['代码'] == code]
            top_list_status = "上榜" if not stock_top.empty else "未上榜"
        else:
            top_list_status = "数据未获取"
            
        # 尝试获取资金流向数据
        try:
            flow_data = get_stock_data_with_retry(
                lambda: ak.stock_fund_flow_individual(symbol=code)
            )
            
            if not flow_data.empty:
                main_net_inflow = flow_data['主力净流入'].iloc[-1]
            else:
                main_net_inflow = "数据未获取"
        except:
            main_net_inflow = "数据未获取"
            
        return {
            'main_net_inflow': main_net_inflow,
            'total_turnover': turnover,
            'top_list_status': top_list_status
        }
        
    except Exception as e:
        return f"获取资金流向失败: {str(e)}"

def get_industry_compare(code):
    """获取行业对比数据"""
    try:
        # 获取市场数据
        market_data = get_batch_market_data()
        
        if not market_data:
            return "获取数据失败"
            
        market_df = market_data.get('market', pd.DataFrame())
        if market_df.empty:
            return "市场数据为空"
            
        # 打印列名以便调试
        st.write("市场数据列名:", list(market_df.columns))
        
        # 获取行业信息
        try:
            # 使用东方财富行业数据
            industry_data = get_stock_data_with_retry(
                lambda: ak.stock_board_industry_cons_em()
            )
            
            if industry_data.empty:
                return "行业数据为空"
                
            # 查找股票所属行业
            stock_industry = industry_data[industry_data['代码'] == code]
            if stock_industry.empty:
                return "未找到股票行业信息"
                
            industry = stock_industry['行业'].iloc[0]
            
            # 获取同行业股票
            industry_stocks = industry_data[industry_data['行业'] == industry]
            industry_codes = industry_stocks['代码'].tolist()
            
            # 获取行业内股票的涨跌幅
            industry_performance = []
            for ind_code in industry_codes:
                stock_data = market_df[market_df['代码'] == ind_code]
                if not stock_data.empty and '涨跌幅' in stock_data.columns:
                    change = stock_data['涨跌幅'].iloc[0]
                    industry_performance.append((ind_code, change))
            
            if not industry_performance:
                return "无法获取行业表现数据"
                
            # 计算行业排名
            industry_performance.sort(key=lambda x: x[1], reverse=True)
            total_stocks = len(industry_performance)
            try:
                current_rank = next(i for i, (ind_code, _) in enumerate(industry_performance) if ind_code == code)
                rank = current_rank + 1
            except StopIteration:
                return "未找到股票排名"
            
            # 计算相对强度
            if rank <= total_stocks * 0.3:
                strength = "强势"
            elif rank <= total_stocks * 0.7:
                strength = "中性"
            else:
                strength = "弱势"
                
            return {
                'industry': industry,
                'rank': rank,
                'total': total_stocks,
                'strength': strength,
                'performance': f"{industry_performance[current_rank][1]:.2f}%"
            }
            
        except Exception as e:
            st.warning(f"处理行业数据时出错: {str(e)}")
            
            # 使用备用方案：从股票名称判断行业
            stock_data = market_df[market_df['代码'] == code]
            if not stock_data.empty:
                stock_name = stock_data['名称'].iloc[0]
                # 可以添加一些简单的行业判断逻辑
                return {
                    'industry': "行业数据获取失败",
                    'rank': "-",
                    'total': "-",
                    'strength': "未知",
                    'performance': stock_data['涨跌幅'].iloc[0]
                }
            
            return "无法获取行业信息"
            
    except Exception as e:
        return f"获取行业对比失败: {str(e)}"

def main():
    """主函数"""
    st.title("事件驱动选股策略")
    
    # 添加策略说明
    st.markdown("""
    ### 策略说明
    1. 基础筛选
       - 剔除ST股票
       - 剔除低成交额股票
       - 剔除亏损股票
       
    2. 技术指标
       - MACD金叉
       - KDJ超卖反转
       - 均线多头排列
       
    3. 市场强度
       - 板块涨幅排名
       - 换手率分析
       
    4. 资金流向
       - 龙虎榜数据
       - 资金活跃度
    """)
    
    # 添加参数设置
    with st.expander("参数设置"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 技术指标参数")
            macd_fast = st.slider("MACD快线周期", 5, 20, 12)
            macd_slow = st.slider("MACD慢线周期", 20, 40, 26)
            macd_signal = st.slider("MACD信号周期", 5, 15, 9)
            
            kdj_n = st.slider("KDJ周期", 5, 20, 9)
            kdj_m1 = st.slider("KDJ M1", 1, 5, 3)
            kdj_m2 = st.slider("KDJ M2", 1, 5, 3)
            
            ma_short = st.slider("短期均线", 5, 20, 5)
            ma_mid = st.slider("中期均线", 20, 40, 20)
            ma_long = st.slider("长期均线", 40, 120, 60)
            
        with col2:
            st.markdown("#### 市场指标参数")
            industry_rank = st.slider("板块排名百分比", 10, 50, 30)
            min_turnover = st.slider("最小换手率", 1.0, 10.0, 3.0)
            min_amount = st.slider("最小成交额(万)", 1000, 10000, 3000)
            
            st.markdown("#### 资金流向参数")
            flow_days = st.slider("资金流向观察天数", 1, 10, 3)
            min_net_inflow = st.slider("最小净流入(万)", 100, 2000, 500)
    
    # 添加缓存控制 - 单行显示
    if st.button("清除缓存", type="primary"):
        st.cache_data.clear()
        st.success("缓存已清除")
    
    if st.button("开始选股", type="primary"):
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
        1. MACD指标
           - DIFF线上穿DEA线（金叉）
           - MACD柱由负转正
           - DIF底背离（可选）
           - 零轴下方金叉更有效
           
        2. KDJ指标
           - K值和D值在超卖区（低于20）
           - J值开始上升（底部反转）
           - K线上穿D线
           - J值底背离（可选）
           
        3. 均线系统
           - 短期均线上穿中期均线
           - 中期均线上穿长期均线
           - 三线维持多头排列
           - 长期均线向上运行
           
        #### 市场强度指标
        1. 行业板块分析
           - 所属板块涨幅排名前30%
           - 板块整体趋势向上
           - 板块轮动初期介入
           
        2. 成交量分析
           - 换手率高于行业均值
           - 量价配合，放量上涨
           - 连续交易日活跃度提升
           
        #### 资金流向分析
        1. 主力资金
           - 主力净流入为正
           - 连续多日累计流入
           - 大单买入占比提升
           
        2. 市场关注度
           - 龙虎榜上榜
           - 机构持仓变动
           - 北向资金动向
           
        #### 风险控制
        1. 止损策略
           - 技术指标背离
           - 跌破重要支撑位
           - 资金流向转负
           
        2. 持仓时间
           - 短线（1-3天）
           - 波段（5-10天）
           - 根据趋势调整
        """)

if __name__ == "__main__":
    main() 