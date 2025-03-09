import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os
import time
from functools import lru_cache

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 设置页面配置
st.set_page_config(
    page_title="A股板块分析",
    page_icon="📊",
    layout="wide"
)

# 禁用代理设置
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# 缓存装饰器
@st.cache_data(ttl=3600)
def get_sector_list():
    """获取A股板块列表"""
    try:
        # 获取东方财富行业板块列表
        sector_df = ak.stock_board_industry_name_em()
        st.write(sector_df)
        return sector_df
    except Exception as e:
        st.error(f"获取板块列表失败: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_sector_history(index_code, start_date="20210101", end_date=None):
    """获取板块历史数据"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    
    try:
        # 获取东方财富行业板块历史数据
        st.info(f"正在获取板块 {index_code} 的历史数据...")
        hist_data = ak.stock_board_industry_hist_em(
            symbol=index_code, 
            start_date=start_date, 
            end_date=end_date
        )
        
        # 调试信息
        if hist_data.empty:
            st.warning(f"板块 {index_code} 返回的历史数据为空")
        else:
            st.success(f"成功获取板块 {index_code} 历史数据，共 {len(hist_data)} 条记录")
            
        return hist_data
    except Exception as e:
        st.error(f"获取板块 {index_code} 历史数据失败: {str(e)}")
        import traceback
        st.error(f"详细错误: {traceback.format_exc()}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_sector_stocks(index_code):
    """获取板块成分股"""
    try:
        # 获取东方财富行业板块成分股
        stocks = ak.stock_board_industry_cons_em(symbol=index_code)
        return stocks
    except Exception as e:
        st.error(f"获取板块成分股失败: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_history(stock_code, start_date="20210101", end_date=None):
    """获取个股历史数据"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    
    try:
        # 获取个股历史数据
        hist_data = ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
        return hist_data
    except Exception as e:
        #st.error(f"获取个股 {stock_code} 历史数据失败: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_funds_by_sector(sector_name):
    """获取与板块相关的基金（基础版）"""
    try:
        # 获取所有基金
        funds = ak.fund_open_fund_rank_em()
        
        # 清理板块名称
        keywords = sector_name.replace("板块", "").replace("概念", "").replace("指数", "")
        
        # 在基金名称中搜索关键词
        matched_funds = funds[funds['基金简称'].str.contains(keywords)]
        
        if matched_funds.empty:
            # 尝试搜索更短的关键词
            if len(keywords) > 2:
                shorter_keyword = keywords[:2]  # 取前两个字符
                matched_funds = funds[funds['基金简称'].str.contains(shorter_keyword)]
        
        return matched_funds
    except Exception as e:
        st.error(f"获取板块相关基金失败: {str(e)}")
        return pd.DataFrame()

def analyze_sector_funds(sector_name):
    """分析板块相关基金"""
    try:
        # 获取相关基金
        funds = get_funds_by_sector(sector_name)
        
        if funds.empty:
            st.warning(f"未找到与 {sector_name} 相关的基金")
            return None
        
        # 筛选列
        if '基金代码' in funds.columns and '基金简称' in funds.columns:
            select_columns = [col for col in ['基金代码', '基金简称', '日增长率', '近1周', '近1月', '近3月', '近6月', '近1年', '今年来', '成立来'] 
                             if col in funds.columns]
            funds = funds[select_columns]
            
            # 转换百分比
            percent_columns = [col for col in ['日增长率', '近1周', '近1月', '近3月', '近6月', '近1年', '今年来', '成立来'] 
                              if col in funds.columns]
            for col in percent_columns:
                funds[col] = pd.to_numeric(funds[col].replace(['', '---'], np.nan), errors='coerce')
            
            # 排序
            if '近1年' in funds.columns:
                funds = funds.sort_values('近1年', ascending=False, na_position='last')
            
            return funds
        else:
            st.warning("获取的基金数据格式不正确")
            return None
    
    except Exception as e:
        st.error(f"分析板块相关基金失败: {str(e)}")
        return None

def calculate_sector_performance(sector_df):
    """计算板块表现"""
    results = []
    
    with st.spinner("正在分析板块表现..."):
        total = len(sector_df)
        progress_bar = st.progress(0)
        
        for i, row in sector_df.iterrows():
            # 更新进度条
            progress_bar.progress((i+1)/total)
            
            sector_code = row['板块代码'] if '板块代码' in row else None
            sector_name = row['板块名称'] if '板块名称' in row else None
            
            if not sector_name:
                continue
                
            # 使用板块名称获取历史数据
            hist_data = get_sector_history(sector_name)
            
            if hist_data.empty:
                st.warning(f"板块 {sector_name} 没有历史数据，跳过")
                continue
                
            try:
                # 计算涨跌幅
                hist_data['日期'] = pd.to_datetime(hist_data['日期'])
                first_close = hist_data.iloc[0]['收盘']
                last_close = hist_data.iloc[-1]['收盘']
                change_pct = (last_close - first_close) / first_close * 100
                
                # 获取成分股
                stocks = get_sector_stocks(sector_name)
                stock_count = len(stocks) if not stocks.empty else 0
                
                # 获取相关基金
                funds = get_funds_by_sector(sector_name)
                fund_count = len(funds) if not funds.empty else 0
                
                results.append({
                    '板块代码': sector_code,
                    '板块名称': sector_name,
                    '起始日期': hist_data['日期'].min().strftime('%Y-%m-%d'),
                    '截止日期': hist_data['日期'].max().strftime('%Y-%m-%d'),
                    '起始价格': first_close,
                    '最新价格': last_close,
                    '涨跌幅(%)': round(change_pct, 2),
                    '成分股数量': stock_count,
                    '相关基金数量': fund_count
                })
            except Exception as e:
                st.error(f"处理板块 {sector_name} 数据时出错: {str(e)}")
                import traceback
                st.error(f"详细错误: {traceback.format_exc()}")
        
        # 完成进度条
        progress_bar.empty()
    
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('涨跌幅(%)', ascending=False)
    return result_df

def analyze_sector_stocks(sector_code, sector_name, start_date="20210101"):
    """分析板块成分股表现"""
    try:
        # 获取成分股
        stocks = get_sector_stocks(sector_name)  # 使用板块名称
        
        if stocks.empty:
            st.warning(f"未找到 {sector_name} 的成分股")
            return None, None
        
        # 计算成分股表现
        stock_results = []
        
        with st.spinner(f"正在分析 {sector_name} 的成分股表现..."):
            total = len(stocks)
            progress_bar = st.progress(0)
            
            for i, stock in enumerate(stocks.itertuples()):
                # 更新进度条
                progress_bar.progress((i+1)/total)
                
                stock_code = stock.代码 if hasattr(stock, '代码') else None
                stock_name = stock.名称 if hasattr(stock, '名称') else None
                stock_weight = stock.权重 if hasattr(stock, '权重') else 0
                
                if not stock_code:
                    continue
                
                # 获取个股历史数据
                hist_data = get_stock_history(stock_code, start_date)
                
                if hist_data.empty:
                    continue
                
                # 计算涨跌幅
                hist_data['日期'] = pd.to_datetime(hist_data['日期'])
                first_close = hist_data.iloc[0]['收盘']
                last_close = hist_data.iloc[-1]['收盘']
                change_pct = (last_close - first_close) / first_close * 100
                
                stock_results.append({
                    '股票代码': stock_code,
                    '股票名称': stock_name,
                    '权重(%)': stock_weight,
                    '起始价格': first_close,
                    '最新价格': last_close,
                    '涨跌幅(%)': round(change_pct, 2)
                })
            
            # 完成进度条
            progress_bar.empty()
        
        stock_df = pd.DataFrame(stock_results)
        if stock_df.empty:
            return None, None
        
        # 按权重排序
        weight_df = stock_df.sort_values('权重(%)', ascending=False).head(10)
        
        # 按涨跌幅排序
        performance_df = stock_df.sort_values('涨跌幅(%)', ascending=False)
        
        return weight_df, performance_df
    
    except Exception as e:
        st.error(f"分析板块成分股失败: {str(e)}")
        return None, None

def plot_sector_performance(sector_df):
    """绘制板块表现图表"""
    # 筛选有数据的前20个板块
    plot_df = sector_df.head(20).copy()
    
    # 创建条形图
    fig = px.bar(
        plot_df,
        x='板块名称',
        y='涨跌幅(%)',
        title='板块涨跌幅排行(2021年至今)',
        color='涨跌幅(%)',
        color_continuous_scale=px.colors.diverging.RdBu_r,
        text='涨跌幅(%)'
    )
    
    fig.update_layout(
        xaxis_title='板块',
        yaxis_title='涨跌幅(%)',
        height=500
    )
    
    return fig

def sector_analysis():
    
    st.title("📊 A股板块分析")
    st.subheader("行业板块表现分析 (2021年至今)")
    
    # 获取板块列表
    with st.spinner("正在获取板块数据..."):
        sector_df = get_sector_list()
    
    if sector_df.empty:
        st.error("获取板块数据失败")
        return
    
    # 计算板块表现
    performance_df = calculate_sector_performance(sector_df)
    
    # 展示板块表现图表
    fig = plot_sector_performance(performance_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # 展示板块表现表格
    st.subheader("板块表现排名")
    st.dataframe(
        performance_df.style.format({
            '涨跌幅(%)': '{:.2f}',
            '起始价格': '{:.2f}',
            '最新价格': '{:.2f}'
        }).bar(
            subset=['涨跌幅(%)'],
            color=['#d65f5f', '#5fba7d']
        ),
        height=400,
        use_container_width=True
    )
    
    # 选择要查看详情的板块
    selected_sector = st.selectbox(
        "选择板块查看详情:",
        performance_df['板块名称'].tolist()
    )
    
    if selected_sector:
        sector_info = performance_df[performance_df['板块名称'] == selected_sector].iloc[0]
        sector_code = sector_info['板块代码']
        
        st.header(f"{selected_sector} 详细分析")
        
        # 显示板块基本信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("板块代码", sector_code)
        with col2:
            st.metric("成分股数量", sector_info['成分股数量'])
        with col3:
            st.metric("总涨跌幅", f"{sector_info['涨跌幅(%)']}%")
        
        # 分析板块成分股
        weight_df, performance_df = analyze_sector_stocks(sector_code, selected_sector)
        
        if weight_df is not None and performance_df is not None:
            # 展示权重最大的10只股票
            st.subheader(f"{selected_sector} 权重最大的10只股票")
            st.dataframe(
                weight_df.style.format({
                    '权重(%)': '{:.2f}',
                    '涨跌幅(%)': '{:.2f}',
                    '起始价格': '{:.2f}',
                    '最新价格': '{:.2f}'
                }).bar(
                    subset=['涨跌幅(%)'],
                    color=['#d65f5f', '#5fba7d']
                ),
                height=400,
                use_container_width=True
            )
            
            # 展示涨跌幅最大的10只股票
            st.subheader(f"{selected_sector} 涨跌幅所有股票")
            st.dataframe(
                performance_df.style.format({
                    '权重(%)': '{:.2f}',
                    '涨跌幅(%)': '{:.2f}',
                    '起始价格': '{:.2f}',
                    '最新价格': '{:.2f}'
                }).bar(
                    subset=['涨跌幅(%)'],
                    color=['#d65f5f', '#5fba7d']
                ),
                height=400,
                use_container_width=True
            )
        
        # 分析板块相关基金
        funds_df = analyze_sector_funds(selected_sector)
        
        if funds_df is not None and not funds_df.empty:
            st.subheader(f"{selected_sector} 相关基金")
            
            # 显示基金表格
            format_cols = {}
            for col in funds_df.columns:
                if col in ['日增长率', '近1周', '近1月', '近3月', '近6月', '近1年', '今年来', '成立来']:
                    format_cols[col] = '{:.2f}'
            
            st.dataframe(
                funds_df.style.format(format_cols),
                height=400,
                use_container_width=True
            )
        else:
            st.info(f"未找到与 {selected_sector} 相关的基金")

if __name__ == "__main__":
    sector_analysis() 
    