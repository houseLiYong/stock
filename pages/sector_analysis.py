import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
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
        return sector_df
    except Exception as e:
        st.error(f"获取板块列表失败: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_sector_history(index_code, start_date, end_date):
    """获取板块历史数据"""
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
    """获取与板块相关的基金（优化关键词版）"""
    try:
        # 获取所有基金
        funds = ak.fund_open_fund_rank_em()
        
        # 行业关键词映射表 - 将板块名称映射到更精确的基金关键词
        industry_mapping = {
            "电子": ["电子", "芯片", "半导体"],
            "医药": ["医药", "医疗", "生物", "医药生物", "医疗器械", "创新药"],
            "食品饮料": ["食品", "饮料", "消费", "白酒"],
            "银行": ["银行", "金融"],
            "计算机": ["计算机", "软件", "信息技术", "IT", "互联网", "人工智能", "云计算"],
            "通信": ["通信", "5G", "移动互联"],
            "汽车": ["汽车", "新能源车", "汽车零部件"],
            "电气设备": ["电气", "设备", "输配电", "电力设备"],
            "机械设备": ["机械", "工业", "装备", "制造"],
            "建筑材料": ["建材", "建筑", "水泥", "装饰"],
            "农林牧渔": ["农业", "农林", "牧渔", "养殖"],
            "基础化工": ["化工", "化学", "材料"],
            "房地产": ["地产", "房地产", "物业"],
            "钢铁": ["钢铁", "金属", "有色", "钢材"],
            "家用电器": ["家电", "消费电子", "智能家居"],
            "商贸零售": ["商贸", "零售", "商业", "电商"],
            "纺织服装": ["纺织", "服装", "服饰", "纺织品"],
            "传媒": ["传媒", "文化", "娱乐", "影视"],
            "国防军工": ["军工", "国防", "航空", "航天"],
            "煤炭": ["煤炭", "能源", "矿业"],
            "美容护理": ["美容", "护理", "化妆品"],
            "电力": ["电力", "公用事业", "火电", "水电"],
            "非银金融": ["证券", "保险", "多元金融"],
            "有色金属": ["有色", "贵金属", "稀有金属", "铜", "金", "铝"],
            "交通运输": ["交通", "运输", "港口", "航运", "铁路", "公路", "物流"],
            "光伏": ["光伏", "太阳能", "新能源"], 
            "风电": ["风电", "风能"],
            "新能源车": ["新能源汽车", "电动车", "汽车新能源"],
            "半导体": ["半导体", "集成电路", "芯片"],
            "新能源": ["新能源", "氢能", "储能", "光伏", "风电"]
        }
        
        # 清理板块名称
        clean_sector = sector_name.replace("板块", "").replace("概念", "").replace("指数", "").replace("行业", "")
        
        # 寻找最匹配的行业类别
        matched_keywords = []
        
        # 1. 直接在映射表中查找完全匹配
        for industry, keywords in industry_mapping.items():
            if clean_sector == industry:
                matched_keywords = keywords
                break
        
        # 2. 如果没有完全匹配，尝试部分匹配
        if not matched_keywords:
            for industry, keywords in industry_mapping.items():
                if industry in clean_sector or clean_sector in industry:
                    matched_keywords = keywords
                    break
        
        # 3. 如果仍未匹配，尝试关键词部分匹配
        if not matched_keywords:
            for industry, keywords in industry_mapping.items():
                for keyword in keywords:
                    if keyword in clean_sector or clean_sector in keyword:
                        matched_keywords = keywords
                        break
                if matched_keywords:
                    break
        
        # 4. 如果上述都未匹配，则使用原始板块名称作为关键词
        if not matched_keywords:
            matched_keywords = [clean_sector]
            # 如果超过3个字符，也添加前2个字符作为辅助关键词
            if len(clean_sector) > 2:
                matched_keywords.append(clean_sector[:2])
        
        # 记录搜索关键词
        st.write(f"搜索关键词: {matched_keywords}")
        
        # 使用所有匹配的关键词搜索基金
        result_funds = pd.DataFrame()
        for keyword in matched_keywords:
            # 防止关键词太短导致过度匹配
            if len(keyword) >= 2:
                # 在基金名称和基金简称中搜索
                matched = funds[funds['基金简称'].str.contains(keyword)]
                if not matched.empty:
                    if result_funds.empty:
                        result_funds = matched
                    else:
                        result_funds = pd.concat([result_funds, matched])
        
        # 去重
        if not result_funds.empty:
            result_funds = result_funds.drop_duplicates(subset=['基金代码'])
        
        # 如果结果太多（>30），尝试使用更严格的匹配条件
        if len(result_funds) > 30:
            strict_results = pd.DataFrame()
            # 使用第一个关键词（通常是最主要的）进行更严格的匹配
            primary_keyword = matched_keywords[0]
            strict_results = funds[funds['基金简称'].str.contains(primary_keyword)]
            
            # 如果严格匹配有结果且数量合理，使用严格匹配结果
            if not strict_results.empty and len(strict_results) < 30:
                result_funds = strict_results
        
        return result_funds
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

def calculate_sector_performance(sector_df, start_date, end_date):
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
            hist_data = get_sector_history(sector_name, start_date, end_date)
            
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
        st.write(stocks)
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
                    '起始价格': first_close,
                    '最新价格': last_close,
                    '涨跌幅(%)': round(change_pct, 2)
                })
            
            # 完成进度条
            progress_bar.empty()
        
        stock_df = pd.DataFrame(stock_results)
        if stock_df.empty:
            return None, None
        
        # 按涨跌幅排序
        performance_df = stock_df.sort_values('涨跌幅(%)', ascending=False)
        
        return  performance_df
    
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
    """板块分析主函数"""
    st.title("A股板块分析")
    
    # 日期选择器
    col1, col2 = st.columns(2)
    with col1:
        # 默认起始日期为一年前
        default_start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        start_date = st.date_input(
            "起始日期", 
            value=datetime.strptime(default_start_date, "%Y-%m-%d"),
            min_value=datetime(2015, 1, 1),
            max_value=datetime.now() - timedelta(days=1)
        )
    with col2:
        end_date = st.date_input(
            "结束日期", 
            value=datetime.now(),
            min_value=datetime(2015, 1, 1),
            max_value=datetime.now()
        )
    
    # 转换日期格式
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    
    # 验证日期范围
    if start_date >= end_date:
        st.error("起始日期必须早于结束日期")
        return
    
    # 介绍性文字
    date_range_text = f"{start_date.strftime('%Y年%m月%d日')} 至 {end_date.strftime('%Y年%m月%d日')}"
    st.markdown(f"### 分析时间范围: {date_range_text}")
    st.markdown("本页面分析A股各个行业板块的表现，展示涨跌幅排行，并提供成分股和相关基金分析。")
    
    # 获取板块列表
    sectors = get_sector_list()
    
    if sectors.empty:
        st.error("无法获取板块列表数据")
        return
    
    # 计算板块表现
    with st.expander("板块涨跌幅排行", expanded=True):
        performance = calculate_sector_performance(sectors, start_date_str, end_date_str)
        
        if performance.empty:
            st.error("无法计算板块表现")
            return
        
        # 显示板块表现
        st.dataframe(
            performance.style.format({
                '涨跌幅(%)': '{:.2f}',
                '起始价格': '{:.2f}',
                '最新价格': '{:.2f}'
            }).background_gradient(
                cmap='RdYlGn',
                subset=['涨跌幅(%)']
            ),
            height=400,
            use_container_width=True
        )
        
        # 可视化前20个板块
        fig = px.bar(
            performance.head(20),
            y='板块名称',
            x='涨跌幅(%)',
            orientation='h',
            title=f"行业板块涨跌幅排行 (TOP 20) - {date_range_text}",
            color='涨跌幅(%)',
            color_continuous_scale='RdYlGn',
            text='涨跌幅(%)'
        )
        
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # 板块详细分析
    st.subheader("板块详细分析")
    selected_sector = st.selectbox("选择板块", performance['板块名称'].tolist())
    
    if selected_sector:
        # 显示板块概况
        sector_info = performance[performance['板块名称'] == selected_sector].iloc[0]
        st.markdown(f"#### {selected_sector} 概况")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("涨跌幅", f"{sector_info['涨跌幅(%)']}%")
        with col2:
            st.metric("起始价格", f"{sector_info['起始价格']}")
        with col3:
            st.metric("最新价格", f"{sector_info['最新价格']}")
        with col4:
            st.metric("成分股数量", f"{sector_info['成分股数量']}")
        
        # 分析板块成分股
        performance_df = analyze_sector_stocks(
            sector_info['板块代码'] if '板块代码' in sector_info else None,
            selected_sector,
            start_date_str
        )
        
        col1, = st.columns(1)
        
        with col1:
            st.markdown(f"#### {selected_sector} 涨跌幅最大的股票")
            if performance_df is not None and not performance_df.empty:
                st.dataframe(
                    performance_df.style.format({
                        '起始价格': '{:.2f}',
                        '最新价格': '{:.2f}',
                        '涨跌幅(%)': '{:.2f}'
                    }).background_gradient(
                        cmap='RdYlGn',
                        subset=['涨跌幅(%)']
                    ),
                    height=400,
                    use_container_width=True
                )
            else:
                st.info(f"未找到 {selected_sector} 的涨跌幅信息")
                
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
    