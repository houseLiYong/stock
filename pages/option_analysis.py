import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def process_single_file(uploaded_file):
    """处理单个文件并返回处理后的数据"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_name = uploaded_file.name.lower().replace('.' + file_extension, '')
        
        if file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            st.warning(f"不支持的文件格式: {uploaded_file.name}")
            return None
            
        # 清理列名
        df.columns = df.columns.str.strip().str.replace('\s+', ' ')
        
        # 根据文件名确定数据类型
        if '沪深300期权' in file_name and '认沽认购' in file_name:
            df = df.rename(columns={df.columns[1]: '沪深300期权_认沽认购持仓比'})
        elif '50etf' in file_name and '认沽认购' in file_name:
            df = df.rename(columns={df.columns[1]: '华夏上证50ETF期权_认沽认购持仓比'})
        elif '50etf' in file_name and '认沽波动率' in file_name:
            df = df.rename(columns={df.columns[1]: '华夏上证50ETF期权_认沽隐含波动率'})
        elif '50etf' in file_name and '认购波动率' in file_name:
            df = df.rename(columns={df.columns[1]: '华夏上证50ETF期权_认购隐含波动率'})
        elif '上证指数' in file_name:
            df = df.rename(columns={df.columns[1]: '上证指数'})
            
        # 确保日期列名为'日期'
        df = df.rename(columns={df.columns[0]: '日期'})
        
        # 转换日期列
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        
        # 删除日期为空的行
        df = df.dropna(subset=['日期'])
        
        # 删除包含特定字符串的行
        invalid_strings = ['通联数据', 'DataYes', '数据来源']
        for invalid_str in invalid_strings:
            df = df[~df.astype(str).apply(lambda x: x.str.contains(invalid_str, na=False)).any(axis=1)]
        
        # 将数值列转换为数值类型
        value_column = df.columns[1]  # 第二列为数值列
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        
        # 删除数值为空的行
        df = df.dropna()
        
        return df
    except Exception as e:
        st.warning(f"处理文件 {uploaded_file.name} 时出现错误: {str(e)}")
        return None

def load_and_process_data(uploaded_files):
    """加载和处理上传的数据文件"""
    try:
        all_data = {}
        
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name.lower()
            
            # 根据文件扩展名选择读取方法
            if file_name.endswith('.csv'):
                # 尝试不同的编码方式读取CSV文件
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"成功使用 {encoding} 编码读取CSV文件: {file_name}")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        st.error(f"读取文件时出错 ({encoding}): {str(e)}")
                        continue
            
            elif file_name.endswith(('.xls', '.xlsx')):
                try:
                    df = pd.read_excel(uploaded_file)
                    st.success(f"成功读取Excel文件: {file_name}")
                except Exception as e:
                    st.error(f"读取Excel文件时出错: {str(e)}")
                    continue
            
            else:
                st.error(f"不支持的文件格式: {file_name}")
                continue
            
            if df is None:
                st.error(f"无法读取文件 {file_name}，请检查文件格式")
                continue
            
            # 数据清理步骤
            try:
                # 删除包含"通联数据"等非数据行
                df = df[~df.iloc[:, 0].astype(str).str.contains('通联|数据|说明|备注', na=False)]
                
                # 确保第一列是日期
                df = df.rename(columns={df.columns[0]: 'date'})
                
                # 尝试转换日期，处理不同的日期格式
                try:
                    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
                except ValueError:
                    try:
                        # 尝试其他常见日期格式
                        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)
                    except ValueError:
                        st.error(f"无法解析日期格式，请检查 {file_name} 的日期列")
                        continue
                
                # 删除重复的日期
                df = df.drop_duplicates(subset=['date'])
                
                # 设置日期索引
                df = df.set_index('date')
                
                # 根据文件名重命名列
                if '沪深300期权' in file_name and '认沽认购' in file_name:
                    df = df.rename(columns={df.columns[0]: '沪深300期权_认沽认购持仓比'})
                elif '50etf' in file_name and '认沽认购' in file_name:
                    df = df.rename(columns={df.columns[0]: '华夏上证50ETF期权_认沽认购持仓比'})
                elif '50etf' in file_name and '认沽波动率' in file_name:
                    df = df.rename(columns={df.columns[0]: '华夏上证50ETF期权_认沽隐含波动率'})
                elif '50etf' in file_name and '认购波动率' in file_name:
                    df = df.rename(columns={df.columns[0]: '华夏上证50ETF期权_认购隐含波动率'})
                elif '上证指数' in file_name:
                    if '成交量' in file_name:
                        df = df.rename(columns={df.columns[0]: '上证指数成交量'})
                    else:
                        df = df.rename(columns={df.columns[0]: '上证指数'})
                
                # 确保数值列为数值类型
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 删除所有空值行
                df = df.dropna()
                
                # 显示数据预览
                st.write(f"\n{file_name} 数据预览：")
                st.write(df.head())
                
                # 存储处理后的数据
                for col in df.columns:
                    all_data[col] = df[col]
                
            except Exception as e:
                st.error(f"处理文件 {file_name} 时出错: {str(e)}")
                continue
        
        # 合并所有数据
        if all_data:
            final_df = pd.DataFrame(all_data)
            
            # 检查是否包含必要的列
            required_columns = [
                '华夏上证50ETF期权_认沽认购持仓比',
                '沪深300期权_认沽认购持仓比',
                '上证指数'
            ]
            
            missing_columns = [col for col in required_columns if col not in final_df.columns]
            if missing_columns:
                st.warning(f"缺少必要的数据列: {', '.join(missing_columns)}")
            
            # 显示数据加载信息
            st.write("已加载的数据列：", list(final_df.columns))
            
            # 如果有成交量数据，确保为数值型
            if '上证指数成交量' in final_df.columns:
                final_df['上证指数成交量'] = pd.to_numeric(final_df['上证指数成交量'], errors='coerce')
                st.success("成功加载上证指数成交量数据")
                
                # 显示成交量数据的基本统计信息
                volume_stats = {
                    '平均成交量': final_df['上证指数成交量'].mean(),
                    '最大成交量': final_df['上证指数成交量'].max(),
                    '最小成交量': final_df['上证指数成交量'].min()
                }
                st.write("成交量数据统计（单位：手）：", volume_stats)
            
            return final_df
            
        else:
            st.error("没有成功加载任何数据")
            return None
            
    except Exception as e:
        st.error(f"数据处理出错: {str(e)}")
        st.error("详细错误信息：")
        import traceback
        st.error(traceback.format_exc())
        return None

def analyze_market_sentiment(df):
    """分析市场情绪并给出建议"""
    signals = []
    latest = df.iloc[-1]
    
    try:
        # 1. 分析认沽认购持仓比
        if '华夏上证50ETF期权_认沽认购持仓比' in df.columns:
            etf50_pc_ratio = latest['华夏上证50ETF期权_认沽认购持仓比']
            if etf50_pc_ratio > 1.2:
                signals.append({
                    "signal": "看跌", 
                    "strength": "中", 
                    "reason": f"50ETF期权认沽认购持仓比为{etf50_pc_ratio:.2f}，大于1.2，表明市场偏向悲观"
                })
            elif etf50_pc_ratio < 0.8:
                signals.append({
                    "signal": "看涨", 
                    "strength": "中", 
                    "reason": f"50ETF期权认沽认购持仓比为{etf50_pc_ratio:.2f}，小于0.8，表明市场偏向乐观"
                })
        
        if '沪深300期权_认沽认购持仓比' in df.columns:
            hs300_pc_ratio = latest['沪深300期权_认沽认购持仓比']
            if hs300_pc_ratio > 1.2:
                signals.append({
                    "signal": "看跌", 
                    "strength": "中", 
                    "reason": f"沪深300期权认沽认购持仓比为{hs300_pc_ratio:.2f}，大于1.2，表明市场偏向悲观"
                })
            elif hs300_pc_ratio < 0.8:
                signals.append({
                    "signal": "看涨", 
                    "strength": "中", 
                    "reason": f"沪深300期权认沽认购持仓比为{hs300_pc_ratio:.2f}，小于0.8, 表明市场偏向乐观"
                })
        
        # 2. 分析波动率偏度
        if ('华夏上证50ETF期权_认沽隐含波动率' in df.columns and 
            '华夏上证50ETF期权_认购隐含波动率' in df.columns):
            vol_skew = (latest['华夏上证50ETF期权_认沽隐含波动率'] - 
                       latest['华夏上证50ETF期权_认购隐含波动率'])
            if vol_skew > 5:
                signals.append({
                    "signal": "看跌", 
                    "strength": "强", 
                    "reason": f"波动率偏度为{vol_skew:.2f}，大于5，表明市场对下跌风险的担忧增加"
                })
            elif vol_skew < -5:
                signals.append({
                    "signal": "看涨", 
                    "strength": "强", 
                    "reason": f"波动率偏度为{vol_skew:.2f}，小于-5，表明市场对上涨的预期增强"
                })
        
        # 3. 分析上证指数趋势（如果有数据）
        if '上证指数' in df.columns:
            # 计算20日均线
            ma20 = df['上证指数'].rolling(window=20).mean()
            current_price = latest['上证指数']
            current_ma20 = ma20.iloc[-1]
            
            if current_price > current_ma20:
                signals.append({
                    "signal": "看涨", 
                    "strength": "中", 
                    "reason": f"上证指数（{current_price:.2f}）位于20日均线（{current_ma20:.2f}）上方，趋势向好"
                })
            elif current_price < current_ma20:
                signals.append({
                    "signal": "看跌", 
                    "strength": "中", 
                    "reason": f"上证指数（{current_price:.2f}）位于20日均线（{current_ma20:.2f}）下方，需要谨慎"
                })
        
        # 如果没有任何信号
        if not signals:
            signals.append({
                "signal": "中性", 
                "strength": "弱", 
                "reason": "当前可用数据不足以做出明确判断"
            })
        
        return signals
        
    except Exception as e:
        st.error(f"市场情绪分析出错: {str(e)}")
        return [{
            "signal": "错误", 
            "strength": "无", 
            "reason": f"分析过程出现错误: {str(e)}"
        }]

def plot_analysis_chart(df):
    """绘制趋势分析图表"""
    try:
        # 创建子图
        fig = make_subplots(
            rows=4, cols=1,  # 增加一行显示成交量
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=('市场趋势分析', '期权持仓比', '波动率', '成交量')
        )
        
        # 1. 上证指数（如果存在）
        if '上证指数' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['上证指数'],
                    name='上证指数',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
        
        # 2. 认沽认购持仓比
        if '华夏上证50ETF期权_认沽认购持仓比' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['华夏上证50ETF期权_认沽认购持仓比'],
                    name='50ETF期权PC比',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
        if '沪深300期权_认沽认购持仓比' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['沪深300期权_认沽认购持仓比'],
                    name='沪深300期权PC比',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        # 3. 隐含波动率
        if '华夏上证50ETF期权_认沽隐含波动率' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['华夏上证50ETF期权_认沽隐含波动率'],
                    name='认沽期权波动率',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )
            
        if '华夏上证50ETF期权_认购隐含波动率' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['华夏上证50ETF期权_认购隐含波动率'],
                    name='认购期权波动率',
                    line=dict(color='orange')
                ),
                row=3, col=1
            )
        
        # 添加成交量图表
        if '上证指数成交量' in df.columns:
            # 计算成交量MA20
            df['成交量MA20'] = df['上证指数成交量'].rolling(window=20).mean()
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['上证指数成交量'],
                    name='成交量',
                    marker_color='lightblue'
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['成交量MA20'],
                    name='成交量MA20',
                    line=dict(color='orange')
                ),
                row=4, col=1
            )
        
        # 更新布局
        fig.update_layout(
            height=1000,  # 增加图表高度
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 更新x轴设置
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            tickformat='%Y-%m-%d',  # 日期格式
            tickangle=45,  # 倾斜角度
            tickmode='auto',
            nticks=20,  # 大约显示的刻度数量
            rangeslider=dict(visible=True),  # 添加日期范围滑块
            rangeselector=dict(  # 添加时间范围选择按钮
                buttons=list([
                    dict(count=1, label="1月", step="month", stepmode="backward"),
                    dict(count=3, label="3月", step="month", stepmode="backward"),
                    dict(count=6, label="6月", step="month", stepmode="backward"),
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(step="all", label="全部")
                ])
            )
        )
        
        # 更新y轴网格
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
        
    except Exception as e:
        st.error(f"绘制图表时出错: {str(e)}")
        return None

def analyze_market_extremes(df):
    """分析市场极值点的指标关系"""
    try:
        if '上证指数' not in df.columns:
            st.warning("缺少上证指数数据，无法进行极值分析")
            return
            
        # 计算上证指数的20日移动平均线和标准差
        df['MA20'] = df['上证指数'].rolling(window=20).mean()
        df['STD20'] = df['上证指数'].rolling(window=20).std()
        
        # 定义极值点（用移动平均线±2倍标准差作为初步判断）
        df['市场状态'] = 'normal'
        df.loc[df['上证指数'] > df['MA20'] + 2 * df['STD20'], '市场状态'] = 'top'
        df.loc[df['上证指数'] < df['MA20'] - 2 * df['STD20'], '市场状态'] = 'bottom'
        
        # 分析各个状态下的指标
        st.subheader("市场极值分析")
        
        # 创建结果DataFrame
        results = []
        
        # 分析顶部
        top_data = df[df['市场状态'] == 'top']
        if not top_data.empty:
            if '华夏上证50ETF期权_认沽认购持仓比' in df.columns:
                # 找到最高点的日期和指数值
                max_index_date = top_data['上证指数'].idxmax()
                max_index_value = top_data.loc[max_index_date, '上证指数']
                
                etf50_pc_top = {
                    '状态': '市场顶部',
                    '指标': '50ETF期权PC比',
                    '日期': max_index_date.strftime('%Y-%m-%d'),
                    '上证指数': f"{max_index_value:.2f}",
                    '最小值': top_data['华夏上证50ETF期权_认沽认购持仓比'].min(),
                    '最大值': top_data['华夏上证50ETF期权_认沽认购持仓比'].max(),
                    '平均值': top_data['华夏上证50ETF期权_认沽认购持仓比'].mean(),
                    '中位数': top_data['华夏上证50ETF期权_认沽认购持仓比'].median()
                }
                results.append(etf50_pc_top)
            
            if '沪深300期权_认沽认购持仓比' in df.columns:
                hs300_pc_top = {
                    '状态': '市场顶部',
                    '指标': '沪深300期权PC比',
                    '日期': max_index_date.strftime('%Y-%m-%d'),
                    '上证指数': f"{max_index_value:.2f}",
                    '最小值': top_data['沪深300期权_认沽认购持仓比'].min(),
                    '最大值': top_data['沪深300期权_认沽认购持仓比'].max(),
                    '平均值': top_data['沪深300期权_认沽认购持仓比'].mean(),
                    '中位数': top_data['沪深300期权_认沽认购持仓比'].median()
                }
                results.append(hs300_pc_top)
        
        # 分析底部
        bottom_data = df[df['市场状态'] == 'bottom']
        if not bottom_data.empty:
            if '华夏上证50ETF期权_认沽认购持仓比' in df.columns:
                # 找到最低点的日期和指数值
                min_index_date = bottom_data['上证指数'].idxmin()
                min_index_value = bottom_data.loc[min_index_date, '上证指数']
                
                etf50_pc_bottom = {
                    '状态': '市场底部',
                    '指标': '50ETF期权PC比',
                    '日期': min_index_date.strftime('%Y-%m-%d'),
                    '上证指数': f"{min_index_value:.2f}",
                    '最小值': bottom_data['华夏上证50ETF期权_认沽认购持仓比'].min(),
                    '最大值': bottom_data['华夏上证50ETF期权_认沽认购持仓比'].max(),
                    '平均值': bottom_data['华夏上证50ETF期权_认沽认购持仓比'].mean(),
                    '中位数': bottom_data['华夏上证50ETF期权_认沽认购持仓比'].median()
                }
                results.append(etf50_pc_bottom)
            
            if '沪深300期权_认沽认购持仓比' in df.columns:
                hs300_pc_bottom = {
                    '状态': '市场底部',
                    '指标': '沪深300期权PC比',
                    '日期': min_index_date.strftime('%Y-%m-%d'),
                    '上证指数': f"{min_index_value:.2f}",
                    '最小值': bottom_data['沪深300期权_认沽认购持仓比'].min(),
                    '最大值': bottom_data['沪深300期权_认沽认购持仓比'].max(),
                    '平均值': bottom_data['沪深300期权_认沽认购持仓比'].mean(),
                    '中位数': bottom_data['沪深300期权_认沽认购持仓比'].median()
                }
                results.append(hs300_pc_bottom)
        
        if results:
            # 创建结果DataFrame
            results_df = pd.DataFrame(results)
            
            # 格式化数值列，保留2位小数
            for col in ['最小值', '最大值', '平均值', '中位数']:
                results_df[col] = results_df[col].round(2)
            
            # 设置列顺序
            columns_order = ['状态', '指标', '日期', '上证指数', '最小值', '最大值', '平均值', '中位数']
            results_df = results_df[columns_order]
            
            # 使用 Streamlit 的 dataframe 显示，并添加样式
            def color_status(val):
                if val == '市场顶部':
                    return 'background-color: rgba(255,0,0,0.1); color: red'
                elif val == '市场底部':
                    return 'background-color: rgba(0,255,0,0.1); color: green'
                return ''
            
            # 应用样式
            styled_df = results_df.style.apply(lambda x: [color_status(x['状态'])] * len(x), axis=1)
            
            # 显示表格
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=150  # 调整高度以适应内容
            )
            
            # 添加分析说明
            st.write("\n### 分析说明")
            st.write("1. 市场顶部判断标准：上证指数高于20日均线+2倍标准差")
            st.write("2. 市场底部判断标准：上证指数低于20日均线-2倍标准差")
            st.write("3. 建议使用区间：")
            st.write("   - 当PC比处于底部区间时，可能预示着市场见底信号")
            st.write("   - 当PC比处于顶部区间时，可能预示着市场见顶信号")
            
            # 绘制散点图显示关系
            fig = go.Figure()
            
            if '华夏上证50ETF期权_认沽认购持仓比' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['上证指数'],
                    y=df['华夏上证50ETF期权_认沽认购持仓比'],
                    mode='markers',
                    name='50ETF期权PC比',
                    marker=dict(
                        color=df['市场状态'].map({'top': 'red', 'bottom': 'green', 'normal': 'blue'}),
                        size=8
                    )
                ))
            
            fig.update_layout(
                title='上证指数与期权PC比关系散点图',
                xaxis_title='上证指数',
                yaxis_title='PC比',
                height=500
            )
            
            st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"极值分析出错: {str(e)}")

def analyze_pc_ratio_extremes(df):
    """分析特定PC比区间对应的市场情况"""
    try:
        results = []
        
        # 分析华夏上证50ETF期权PC比
        if '华夏上证50ETF期权_认沽认购持仓比' in df.columns:
            # 低位区间 (< 70)
            low_pc = df[df['华夏上证50ETF期权_认沽认购持仓比'] < 70]
            if not low_pc.empty:
                for idx, row in low_pc.iterrows():
                    results.append({
                        '日期': idx.strftime('%Y-%m-%d'),
                        '指标': '50ETF期权PC比',
                        'PC比值': round(row['华夏上证50ETF期权_认沽认购持仓比'], 2),
                        '上证指数': round(row['上证指数'], 2),
                        '区间': '低位区间(<70)'
                    })
            
            # 高位区间 (> 100)
            high_pc = df[df['华夏上证50ETF期权_认沽认购持仓比'] > 100]
            if not high_pc.empty:
                for idx, row in high_pc.iterrows():
                    results.append({
                        '日期': idx.strftime('%Y-%m-%d'),
                        '指标': '50ETF期权PC比',
                        'PC比值': round(row['华夏上证50ETF期权_认沽认购持仓比'], 2),
                        '上证指数': round(row['上证指数'], 2),
                        '区间': '高位区间(>100)'
                    })
        
        # 分析沪深300期权PC比
        if '沪深300期权_认沽认购持仓比' in df.columns:
            # 低位区间 (< 60)
            low_pc = df[df['沪深300期权_认沽认购持仓比'] < 60]
            if not low_pc.empty:
                for idx, row in low_pc.iterrows():
                    results.append({
                        '日期': idx.strftime('%Y-%m-%d'),
                        '指标': '沪深300期权PC比',
                        'PC比值': round(row['沪深300期权_认沽认购持仓比'], 2),
                        '上证指数': round(row['上证指数'], 2),
                        '区间': '低位区间(<60)'
                    })
            
            # 高位区间 (> 85)
            high_pc = df[df['沪深300期权_认沽认购持仓比'] > 85]
            if not high_pc.empty:
                for idx, row in high_pc.iterrows():
                    results.append({
                        '日期': idx.strftime('%Y-%m-%d'),
                        '指标': '沪深300期权PC比',
                        'PC比值': round(row['沪深300期权_认沽认购持仓比'], 2),
                        '上证指数': round(row['上证指数'], 2),
                        '区间': '高位区间(>85)'
                    })
        
        if results:
            # 创建DataFrame并排序
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('日期', ascending=False)
            
            # 设置样式
            def color_range(val):
                if '高位' in val:
                    return 'background-color: rgba(255,0,0,0.1); color: red'
                elif '低位' in val:
                    return 'background-color: rgba(0,255,0,0.1); color: green'
                return ''
            
            # 应用样式
            styled_df = results_df.style.apply(lambda x: [color_range(x['区间'])] * len(x), axis=1)
            
            # 显示结果
            st.subheader("PC比极值区间分析")
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400  # 可根据实际数据量调整
            )
            
            # 添加统计信息
            st.write("\n### 统计信息")
            stats = results_df.groupby(['指标', '区间']).agg({
                'PC比值': ['count', 'mean', 'min', 'max'],
                '上证指数': ['mean', 'min', 'max']
            }).round(2)
            
            st.dataframe(stats)
            
        else:
            st.info("没有找到符合条件的极值数据")
            
    except Exception as e:
        st.error(f"PC比极值分析出错: {str(e)}")

def analyze_combined_signals(df):
    """分析PC比、成交量、IV和IV Skew的组合信号"""
    try:
        # 首先打印所有可用的列名，以便调试
        st.write("可用的数据列：", list(df.columns))
        
        # 检查必要的数据列
        required_columns = [
            '华夏上证50ETF期权_认沽认购持仓比',
            '沪深300期权_认沽认购持仓比',
            '上证指数',
            '上证指数成交量',
            '华夏上证50ETF期权:认沽隐含波动率',
            '华夏上证50ETF期权:认购隐含波动率'
        ]
        st.dataframe(df.columns)
        
        # 检查必要列
        missing_columns = [col for col in required_columns if col not in df.columns]
        st.write(missing_columns)
        if missing_columns:
            st.warning(f"缺少必要的数据列-----: {', '.join(missing_columns)}")
            return
        
        # 计算IV相关指标
        df['IV_Skew'] = df['华夏上证50ETF期权:认沽隐含波动率'] - df['华夏上证50ETF期权:认购隐含波动率']
        df['平均IV'] = (df['华夏上证50ETF期权:认沽隐含波动率'] + df['华夏上证50ETF期权:认购隐含波动率']) / 2
        
        # 计算IV分位数
        iv_90 = df['平均IV'].quantile(0.90)
        iv_50 = df['平均IV'].quantile(0.50)
        
        # 计算成交量指标
        df['成交量MA20'] = df['上证指数成交量'].rolling(window=20).mean()
        df['量能比'] = df['上证指数成交量'] / df['成交量MA20']
        
        results = []
        
        # 找出组合信号
        for date in df.index:
            # 基础信号判断
            pc_50etf = df.loc[date, '华夏上证50ETF期权_认沽认购持仓比']
            pc_hs300 = df.loc[date, '沪深300期权_认沽认购持仓比']
            avg_iv = df.loc[date, '平均IV']
            iv_skew = df.loc[date, 'IV_Skew']
            volume_ratio = df.loc[date, '量能比']
            
            # 判断信号类型
            signal_type = None
            if pc_50etf < 70 and pc_hs300 < 60:
                signal_type = '双低信号'
            elif pc_50etf > 100 and pc_hs300 > 85:
                signal_type = '双高信号'
            
            if signal_type:
                # IV水平判断
                if avg_iv > iv_90:
                    iv_level = '高IV'
                elif avg_iv < iv_50:
                    iv_level = '低IV'
                else:
                    iv_level = '正常IV'
                
                # IV Skew判断
                if iv_skew > 5:
                    skew_type = '看跌偏斜'
                elif iv_skew < -5:
                    skew_type = '看涨偏斜'
                else:
                    skew_type = '中性'
                
                # 量能判断
                volume_strength = '强' if volume_ratio > 1.2 else '中' if volume_ratio > 0.8 else '弱'
                
                # 综合信号强度判断
                signal_strength = '强'
                if signal_type == '双高信号':
                    if iv_level == '高IV' and skew_type == '看跌偏斜' and volume_strength in ['强', '中']:
                        signal_strength = '强'
                    elif volume_strength == '弱' or iv_level == '低IV':
                        signal_strength = '弱'
                    else:
                        signal_strength = '中'
                else:  # 双低信号
                    if iv_level == '低IV' and skew_type == '看涨偏斜' and volume_strength in ['强', '中']:
                        signal_strength = '强'
                    elif volume_strength == '弱' or iv_level == '高IV':
                        signal_strength = '弱'
                    else:
                        signal_strength = '中'
                
                results.append({
                    '日期': date.strftime('%Y-%m-%d'),
                    '信号类型': signal_type,
                    '50ETF期权PC比': round(pc_50etf, 2),
                    '沪深300期权PC比': round(pc_hs300, 2),
                    '上证指数': round(df.loc[date, '上证指数'], 2),
                    '成交量': round(df.loc[date, '上证指数成交量'] / 100000000, 2),
                    '量能比': round(volume_ratio, 2),
                    'IV水平': iv_level,
                    '平均IV': round(avg_iv, 2),
                    'IV Skew': round(iv_skew, 2),
                    'Skew类型': skew_type,
                    '信号强度': signal_strength
                })
        
        if results:
            # 创建DataFrame并按日期降序排序
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('日期', ascending=False)
            
            # 设置样式函数
            def style_row(row):
                styles = pd.Series('', index=row.index)
                
                # 信号类型样式
                if row['信号类型'] == '双高信号':
                    styles['信号类型'] = 'background-color: rgba(255,0,0,0.1)'
                elif row['信号类型'] == '双低信号':
                    styles['信号类型'] = 'background-color: rgba(0,255,0,0.1)'
                
                # IV水平样式
                if row['IV水平'] == '高IV':
                    styles['IV水平'] = 'color: red'
                elif row['IV水平'] == '低IV':
                    styles['IV水平'] = 'color: green'
                
                # Skew类型样式
                if row['Skew类型'] == '看跌偏斜':
                    styles['Skew类型'] = 'color: red'
                elif row['Skew类型'] == '看涨偏斜':
                    styles['Skew类型'] = 'color: green'
                
                # 信号强度样式
                if row['信号强度'] == '强':
                    styles['信号强度'] = 'font-weight: bold'
                elif row['信号强度'] == '弱':
                    styles['信号强度'] = 'color: gray'
                
                return styles
            
            # 应用样式
            styled_df = results_df.style.apply(style_row, axis=1)
            
            # 显示结果
            st.subheader("市场综合信号分析")
            
            # 显示信号说明
            st.write("### 信号判断标准")
            st.write("""
            1. PC比信号：
               - 双低信号：50ETF PC比<70 且 沪深300 PC比<60
               - 双高信号：50ETF PC比>100 且 沪深300 PC比>85
            
            2. IV水平：
               - 高IV：高于90%分位数
               - 低IV：低于50%分位数
               - 正常IV：介于两者之间
            
            3. IV Skew：
               - 看跌偏斜：Skew > 5
               - 看涨偏斜：Skew < -5
               - 中性：介于两者之间
            
            4. 量能比：
               - 强：>1.2
               - 中：0.8-1.2
               - 弱：<0.8
            """)
            
            st.write("### 最强信号组合")
            st.write("""
            1. 最强看跌信号：
               - 双高PC比 + 高IV + 看跌偏斜 + 强/中量能
               - 市场含义：市场恐慌情绪浓厚，避险需求强烈
               - 交易启示：市场下跌趋势明确，可考虑做空或持币观望
            
            2. 最强看涨信号：
               - 双低PC比 + 低IV + 看涨偏斜 + 强/中量能
               - 市场含义：市场情绪稳定，看涨预期增强
               - 交易启示：市场可能进入上涨趋势，可逐步布局多头
            
            3. 信号组合解读：
               - PC比：反映市场看跌/看涨期权的持仓偏好
               - IV水平：反映市场对波动率的预期
               - IV Skew：反映市场的方向性偏好
               - 量能：反映市场交易活跃度
               
            4. 注意事项：
               - 信号仅供参考，需结合其他技术指标
               - 建议结合基本面和市场环境综合判断
               - 极端行情下信号可能失效
               - 请注意仓位控制和风险管理
            """)
            
            # 显示数据表格
            st.write("### 信号明细")
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400
            )
            
            # 统计分析
            st.write("### 信号统计")
            
            # 按信号类型和强度的分布
            signal_dist = pd.crosstab(
                results_df['信号类型'], 
                [results_df['IV水平'], results_df['Skew类型']]
            )
            st.write("信号类型与IV/Skew组合分布：")
            st.dataframe(signal_dist)
            
            # 最新信号分析
            latest = results_df.iloc[0]
            st.write(f"\n### 最新信号 ({latest['日期']})")
            st.info(f"""
            - 信号类型：{latest['信号类型']}
            - IV水平：{latest['IV水平']} (IV={latest['平均IV']:.2f})
            - Skew类型：{latest['Skew类型']} (Skew={latest['IV Skew']:.2f})
            - 量能：{latest['信号强度']} (量能比={latest['量能比']:.2f})
            - 综合强度：{latest['信号强度']}
            """)
            
        else:
            st.info("没有找到符合条件的信号")
            
    except Exception as e:
        st.error(f"组合信号分析出错: {str(e)}")
        st.error("详细错误信息：")
        import traceback
        st.error(traceback.format_exc())

def analyze_indicator_effectiveness(df):
    """分析指标有效性"""
    try:
        st.subheader("指标预测能力分析")
        
        # 计算上证指数的变化
        df['指数变化率_5日'] = df['上证指数'].pct_change(periods=5) * 100  # 5日变化率
        df['指数变化率_10日'] = df['上证指数'].pct_change(periods=10) * 100  # 10日变化率
        df['指数变化率_20日'] = df['上证指数'].pct_change(periods=20) * 100  # 20日变化率
        
        results = []
        
        # 分析50ETF期权PC比
        if '华夏上证50ETF期权_认沽认购持仓比' in df.columns:
            # 低位信号（<70）后的市场表现
            low_signals = df[df['华夏上证50ETF期权_认沽认购持仓比'] < 70]
            if not low_signals.empty:
                results.append({
                    '指标': '50ETF期权PC比',
                    '信号类型': '低位(<70)',
                    '信号次数': len(low_signals),
                    '5日后上涨概率': (low_signals['指数变化率_5日'].shift(-5) > 0).mean() * 100,
                    '10日后上涨概率': (low_signals['指数变化率_10日'].shift(-10) > 0).mean() * 100,
                    '20日后上涨概率': (low_signals['指数变化率_20日'].shift(-20) > 0).mean() * 100,
                    '5日后平均涨跌幅': low_signals['指数变化率_5日'].shift(-5).mean(),
                    '10日后平均涨跌幅': low_signals['指数变化率_10日'].shift(-10).mean(),
                    '20日后平均涨跌幅': low_signals['指数变化率_20日'].shift(-20).mean()
                })
            
            # 高位信号（>100）后的市场表现
            high_signals = df[df['华夏上证50ETF期权_认沽认购持仓比'] > 100]
            if not high_signals.empty:
                results.append({
                    '指标': '50ETF期权PC比',
                    '信号类型': '高位(>100)',
                    '信号次数': len(high_signals),
                    '5日后下跌概率': (high_signals['指数变化率_5日'].shift(-5) < 0).mean() * 100,
                    '10日后下跌概率': (high_signals['指数变化率_10日'].shift(-10) < 0).mean() * 100,
                    '20日后下跌概率': (high_signals['指数变化率_20日'].shift(-20) < 0).mean() * 100,
                    '5日后平均涨跌幅': high_signals['指数变化率_5日'].shift(-5).mean(),
                    '10日后平均涨跌幅': high_signals['指数变化率_10日'].shift(-10).mean(),
                    '20日后平均涨跌幅': high_signals['指数变化率_20日'].shift(-20).mean()
                })
        
        # 分析沪深300期权PC比
        if '沪深300期权_认沽认购持仓比' in df.columns:
            # 低位信号（<60）后的市场表现
            low_signals = df[df['沪深300期权_认沽认购持仓比'] < 60]
            if not low_signals.empty:
                results.append({
                    '指标': '沪深300期权PC比',
                    '信号类型': '低位(<60)',
                    '信号次数': len(low_signals),
                    '5日后上涨概率': (low_signals['指数变化率_5日'].shift(-5) > 0).mean() * 100,
                    '10日后上涨概率': (low_signals['指数变化率_10日'].shift(-10) > 0).mean() * 100,
                    '20日后上涨概率': (low_signals['指数变化率_20日'].shift(-20) > 0).mean() * 100,
                    '5日后平均涨跌幅': low_signals['指数变化率_5日'].shift(-5).mean(),
                    '10日后平均涨跌幅': low_signals['指数变化率_10日'].shift(-10).mean(),
                    '20日后平均涨跌幅': low_signals['指数变化率_20日'].shift(-20).mean()
                })
            
            # 高位信号（>85）后的市场表现
            high_signals = df[df['沪深300期权_认沽认购持仓比'] > 85]
            if not high_signals.empty:
                results.append({
                    '指标': '沪深300期权PC比',
                    '信号类型': '高位(>85)',
                    '信号次数': len(high_signals),
                    '5日后下跌概率': (high_signals['指数变化率_5日'].shift(-5) < 0).mean() * 100,
                    '10日后下跌概率': (high_signals['指数变化率_10日'].shift(-10) < 0).mean() * 100,
                    '20日后下跌概率': (high_signals['指数变化率_20日'].shift(-20) < 0).mean() * 100,
                    '5日后平均涨跌幅': high_signals['指数变化率_5日'].shift(-5).mean(),
                    '10日后平均涨跌幅': high_signals['指数变化率_10日'].shift(-10).mean(),
                    '20日后平均涨跌幅': high_signals['指数变化率_20日'].shift(-20).mean()
                })
        
        if results:
            # 创建DataFrame并显示
            results_df = pd.DataFrame(results)
            
            # 格式化数值列
            for col in results_df.columns:
                if '概率' in col or '涨跌幅' in col:
                    results_df[col] = results_df[col].round(2)
            
            # 设置样式
            def color_signal(val):
                if '高位' in str(val):
                    return 'background-color: rgba(255,0,0,0.1); color: red'
                elif '低位' in str(val):
                    return 'background-color: rgba(0,255,0,0.1); color: green'
                return ''
            
            # 应用样式
            styled_df = results_df.style.apply(lambda x: [color_signal(x['信号类型'])] * len(x), axis=1)
            
            st.dataframe(styled_df, use_container_width=True)
            
            # 分析结论
            st.write("\n### 分析结论")
            
            # 计算预测准确率
            for indicator in ['50ETF期权PC比', '沪深300期权PC比']:
                low_accuracy = results_df[
                    (results_df['指标'] == indicator) & 
                    (results_df['信号类型'].str.contains('低位'))
                ]['10日后上涨概率'].values[0]
                
                high_accuracy = results_df[
                    (results_df['指标'] == indicator) & 
                    (results_df['信号类型'].str.contains('高位'))
                ]['10日后下跌概率'].values[0]
                
                st.write(f"\n**{indicator}分析：**")
                st.write(f"- 低位信号10日后上涨概率: {low_accuracy:.2f}%")
                st.write(f"- 高位信号10日后下跌概率: {high_accuracy:.2f}%")
                st.write(f"- 综合准确率: {((low_accuracy + high_accuracy) / 2):.2f}%")
        
        else:
            st.info("没有足够的数据进行分析")
            
    except Exception as e:
        st.error(f"指标预测能力分析出错: {str(e)}")

def analyze_pcr_ranges(df):
    """分析不同PCR区间的预测能力"""
    try:
        st.subheader("PCR指标区间分析")
        
        # 计算上证指数的变化
        df['指数变化率_5日'] = df['上证指数'].pct_change(periods=5) * 100
        df['指数变化率_10日'] = df['上证指数'].pct_change(periods=10) * 100
        df['指数变化率_20日'] = df['上证指数'].pct_change(periods=20) * 100
        
        # 定义要分析的区间
        ranges_50etf = [
            (0, 50), (50, 60), (60, 70), (70, 80), (80, 90), 
            (90, 100), (100, 110), (110, 120), (120, 130),
            (130, 140), (140, 150), (150, 160), (160, float('inf'))
        ]
        
        ranges_hs300 = [
            (0, 40), (40, 50), (50, 60), (60, 70), (70, 80), 
            (80, 90), (90, 100), (100, 110), (110, 120),
            (120, 130), (130, 140), (140, float('inf'))
        ]
        
        results = []
        
        # 分析50ETF期权PC比
        if '华夏上证50ETF期权_认沽认购持仓比' in df.columns:
            for low, high in ranges_50etf:
                mask = (df['华夏上证50ETF期权_认沽认购持仓比'] >= low)
                if high != float('inf'):
                    mask &= (df['华夏上证50ETF期权_认沽认购持仓比'] < high)
                
                range_data = df[mask]
                if len(range_data) > 0:
                    results.append({
                        '指标': '50ETF期权PC比',
                        '区间': f"{low}-{high if high != float('inf') else '∞'}",
                        '样本数': len(range_data),
                        '5日后上涨概率': (range_data['指数变化率_5日'].shift(-5) > 0).mean() * 100,
                        '10日后上涨概率': (range_data['指数变化率_10日'].shift(-10) > 0).mean() * 100,
                        '20日后上涨概率': (range_data['指数变化率_20日'].shift(-20) > 0).mean() * 100,
                        '5日后平均涨跌幅': range_data['指数变化率_5日'].shift(-5).mean(),
                        '10日后平均涨跌幅': range_data['指数变化率_10日'].shift(-10).mean(),
                        '20日后平均涨跌幅': range_data['指数变化率_20日'].shift(-20).mean()
                    })
        
        # 分析沪深300期权PC比
        if '沪深300期权_认沽认购持仓比' in df.columns:
            for low, high in ranges_hs300:
                mask = (df['沪深300期权_认沽认购持仓比'] >= low)
                if high != float('inf'):
                    mask &= (df['沪深300期权_认沽认购持仓比'] < high)
                
                range_data = df[mask]
                if len(range_data) > 0:
                    results.append({
                        '指标': '沪深300期权PC比',
                        '区间': f"{low}-{high if high != float('inf') else '∞'}",
                        '样本数': len(range_data),
                        '5日后上涨概率': (range_data['指数变化率_5日'].shift(-5) > 0).mean() * 100,
                        '10日后上涨概率': (range_data['指数变化率_10日'].shift(-10) > 0).mean() * 100,
                        '20日后上涨概率': (range_data['指数变化率_20日'].shift(-20) > 0).mean() * 100,
                        '5日后平均涨跌幅': range_data['指数变化率_5日'].shift(-5).mean(),
                        '10日后平均涨跌幅': range_data['指数变化率_10日'].shift(-10).mean(),
                        '20日后平均涨跌幅': range_data['指数变化率_20日'].shift(-20).mean()
                    })
        
        if results:
            # 创建DataFrame并显示
            results_df = pd.DataFrame(results)
            
            # 格式化数值列
            for col in results_df.columns:
                if '概率' in col or '涨跌幅' in col:
                    results_df[col] = results_df[col].round(2)
            
            # 设置样式
            def highlight_extreme_values(s):
                is_numeric = pd.to_numeric(s, errors='coerce').notnull()
                if not is_numeric.any():
                    return [''] * len(s)
                
                if '概率' in s.name:
                    return ['background-color: rgba(0,255,0,0.1)' if v > 60 else
                           'background-color: rgba(255,0,0,0.1)' if v < 40 else
                           '' for v in s]
                elif '涨跌幅' in s.name:
                    return ['background-color: rgba(0,255,0,0.1)' if v > 1 else
                           'background-color: rgba(255,0,0,0.1)' if v < -1 else
                           '' for v in s]
                return [''] * len(s)
            
            # 应用样式
            styled_df = results_df.style.apply(highlight_extreme_values)
            
            # 显示分析结果
            st.write("各区间预测能力分析：")
            st.dataframe(styled_df, use_container_width=True)
            
            # 找出最佳预测区间
            st.write("\n### 最佳预测区间")
            
            for indicator in ['50ETF期权PC比', '沪深300期权PC比']:
                indicator_data = results_df[results_df['指标'] == indicator]
                
                # 找出上涨概率最高的区间
                best_up = indicator_data.nlargest(1, '10日后上涨概率')
                # 找出下跌概率最高的区间（上涨概率最低的区间）
                best_down = indicator_data.nsmallest(1, '10日后上涨概率')
                
                st.write(f"\n**{indicator}最佳区间：**")
                st.write(f"- 看涨信号区间: {best_up['区间'].values[0]} (10日后上涨概率: {best_up['10日后上涨概率'].values[0]:.2f}%)")
                st.write(f"- 看跌信号区间: {best_down['区间'].values[0]} (10日后下跌概率: {100-best_down['10日后上涨概率'].values[0]:.2f}%)")
        
        else:
            st.info("没有足够的数据进行分析")
            
    except Exception as e:
        st.error(f"PCR区间分析出错: {str(e)}")

def analyze_iv_data(df):
    """分析隐含波动率数据"""
    try:
        # 首先检查数据列
        st.write("可用的数据列：", list(df.columns))
        
        # 检查必要的列是否存在
        required_columns = [
            '华夏上证50ETF期权_认沽隐含波动率',
            '华夏上证50ETF期权_认购隐含波动率'
        ]
        
        # 尝试匹配列名（不区分大小写，允许部分匹配）
        iv_columns = {
            'put_iv': None,
            'call_iv': None
        }
        
        for col in df.columns:
            col_lower = col.lower()
            if '认沽' in col_lower and '波动率' in col_lower:
                iv_columns['put_iv'] = col
            elif '认购' in col_lower and '波动率' in col_lower:
                iv_columns['call_iv'] = col
        
        # 显示找到的列
        st.write("找到的IV列：", iv_columns)
        
        # 检查是否找到所需的列
        if not iv_columns['put_iv'] or not iv_columns['call_iv']:
            st.warning("缺少必要的隐含波动率数据列，请检查数据。")
            st.warning("需要包含认沽和认购的隐含波动率数据。")
            return
        
        # 使用找到的列名进行计算
        df['IV_Skew'] = df[iv_columns['put_iv']] - df[iv_columns['call_iv']]
        
        # 计算历史分位数
        put_iv_90 = df[iv_columns['put_iv']].quantile(0.90)
        put_iv_50 = df[iv_columns['put_iv']].quantile(0.50)
        call_iv_90 = df[iv_columns['call_iv']].quantile(0.90)
        call_iv_50 = df[iv_columns['call_iv']].quantile(0.50)
        
        # 创建信号DataFrame
        iv_signals = pd.DataFrame(index=df.index)
        
        # 标记Put IV信号
        iv_signals['认沽IV信号'] = 'normal'
        iv_signals.loc[df[iv_columns['put_iv']] > put_iv_90, '认沽IV信号'] = 'high'
        iv_signals.loc[df[iv_columns['put_iv']] < put_iv_50, '认沽IV信号'] = 'low'
        
        # 标记Call IV信号
        iv_signals['认购IV信号'] = 'normal'
        iv_signals.loc[df[iv_columns['call_iv']] > call_iv_90, '认购IV信号'] = 'high'
        iv_signals.loc[df[iv_columns['call_iv']] < call_iv_50, '认购IV信号'] = 'low'
        
        # 标记IV Skew信号
        iv_signals['IV_Skew信号'] = 'neutral'
        iv_signals.loc[df['IV_Skew'] > 0, 'IV_Skew信号'] = 'bearish'
        iv_signals.loc[df['IV_Skew'] < 0, 'IV_Skew信号'] = 'bullish'
        
        # 显示IV分析结果
        st.subheader("隐含波动率分析")
        
        # 显示当前IV状态
        latest_date = df.index.max()
        latest_data = df.loc[latest_date]
        
        st.write("### 当前IV状态（{}）".format(latest_date.strftime('%Y-%m-%d')))
        current_status = pd.DataFrame({
            '指标': ['认沽IV', '认购IV', 'IV Skew'],
            '当前值': [
                round(latest_data[iv_columns['put_iv']], 2),
                round(latest_data[iv_columns['call_iv']], 2),
                round(latest_data['IV_Skew'], 2)
            ],
            '信号': [
                iv_signals.loc[latest_date, '认沽IV信号'],
                iv_signals.loc[latest_date, '认购IV信号'],
                iv_signals.loc[latest_date, 'IV_Skew信号']
            ]
        })
        st.dataframe(current_status, use_container_width=True)
        
        # 显示历史分位数
        st.write("### IV历史分位数")
        percentiles = pd.DataFrame({
            '指标': ['认沽IV', '认购IV'],
            '90%分位数': [round(put_iv_90, 2), round(call_iv_90, 2)],
            '50%分位数': [round(put_iv_50, 2), round(call_iv_50, 2)]
        })
        st.dataframe(percentiles, use_container_width=True)
        
        # 绘制IV走势图
        fig = go.Figure()
        
        # 添加Put IV
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[iv_columns['put_iv']],
            name='认沽IV',
            line=dict(color='red')
        ))
        
        # 添加Call IV
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[iv_columns['call_iv']],
            name='认购IV',
            line=dict(color='green')
        ))
        
        # 添加IV Skew
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['IV_Skew'],
            name='IV Skew',
            line=dict(color='blue', dash='dash')
        ))
        
        # 添加分位数参考线
        fig.add_hline(y=put_iv_90, line_dash="dot", line_color="red", annotation_text="认沽IV 90%分位")
        fig.add_hline(y=put_iv_50, line_dash="dot", line_color="red", annotation_text="认沽IV 50%分位")
        fig.add_hline(y=call_iv_90, line_dash="dot", line_color="green", annotation_text="认购IV 90%分位")
        fig.add_hline(y=call_iv_50, line_dash="dot", line_color="green", annotation_text="认购IV 50%分位")
        
        fig.update_layout(
            title='隐含波动率走势',
            xaxis_title='日期',
            yaxis_title='IV',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 统计分析
        st.write("### IV信号统计")
        
        # IV状态统计
        def format_date(date):
            return date.strftime('%Y-%m-%d')
        
        # 收集认沽IV信号日期
        put_high_dates = sorted(iv_signals[iv_signals['认沽IV信号'] == 'high'].index, reverse=True)
        put_normal_dates = sorted(iv_signals[iv_signals['认沽IV信号'] == 'normal'].index, reverse=True)
        put_low_dates = sorted(iv_signals[iv_signals['认沽IV信号'] == 'low'].index, reverse=True)
        
        # 收集认购IV信号日期
        call_high_dates = sorted(iv_signals[iv_signals['认购IV信号'] == 'high'].index, reverse=True)
        call_normal_dates = sorted(iv_signals[iv_signals['认购IV信号'] == 'normal'].index, reverse=True)
        call_low_dates = sorted(iv_signals[iv_signals['认购IV信号'] == 'low'].index, reverse=True)
        
        # 创建统计摘要
        iv_summary = pd.DataFrame({
            '信号类型': ['高IV天数', '正常IV天数', '低IV天数'],
            '认沽期权天数': [
                len(put_high_dates),
                len(put_normal_dates),
                len(put_low_dates)
            ],
            '认购期权天数': [
                len(call_high_dates),
                len(call_normal_dates),
                len(call_low_dates)
            ]
        })
        
        # 创建详细日期表格
        def create_date_df(dates, signal_type, option_type):
            if not dates:
                return pd.DataFrame()
            return pd.DataFrame({
                '日期': [format_date(d) for d in dates],
                '信号类型': [signal_type] * len(dates),
                '期权类型': [option_type] * len(dates),
                'IV值': [df.loc[d, iv_columns['put_iv' if option_type == '认沽' else 'call_iv']] for d in dates]
            })
        
        # 合并所有日期数据
        date_dfs = []
        for dates, signal_type, option_type in [
            (put_high_dates, '高IV', '认沽'),
            (put_normal_dates, '正常IV', '认沽'),
            (put_low_dates, '低IV', '认沽'),
            (call_high_dates, '高IV', '认购'),
            (call_normal_dates, '正常IV', '认购'),
            (call_low_dates, '低IV', '认购')
        ]:
            date_dfs.append(create_date_df(dates, signal_type, option_type))
        
        detailed_dates_df = pd.concat(date_dfs, ignore_index=True)
        
        # 使用选项卡显示统计结果
        st.write("#### IV状态统计")
        tab1, tab2 = st.tabs(["统计数据", "详细日期"])
        
        with tab1:
            st.dataframe(iv_summary, use_container_width=True)
        
        with tab2:
            # 添加筛选选项
            col1, col2 = st.columns(2)
            with col1:
                selected_type = st.selectbox(
                    '选择期权类型',
                    ['全部', '认沽', '认购']
                )
            with col2:
                selected_signal = st.selectbox(
                    '选择信号类型',
                    ['全部', '高IV', '正常IV', '低IV']
                )
            
            # 筛选数据
            filtered_df = detailed_dates_df.copy()
            if selected_type != '全部':
                filtered_df = filtered_df[filtered_df['期权类型'] == selected_type]
            if selected_signal != '全部':
                filtered_df = filtered_df[filtered_df['信号类型'] == selected_signal]
            
            # 显示筛选后的数据
            if not filtered_df.empty:
                st.dataframe(
                    filtered_df.sort_values('日期', ascending=False),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("没有符合条件的数据")
        
        # Skew信号统计
        st.write("#### IV Skew统计")
        
        # 收集Skew信号日期
        bearish_dates = sorted(iv_signals[iv_signals['IV_Skew信号'] == 'bearish'].index, reverse=True)
        neutral_dates = sorted(iv_signals[iv_signals['IV_Skew信号'] == 'neutral'].index, reverse=True)
        bullish_dates = sorted(iv_signals[iv_signals['IV_Skew信号'] == 'bullish'].index, reverse=True)
        
        # 创建Skew统计摘要
        skew_summary = pd.DataFrame({
            'Skew类型': ['看跌偏斜(正Skew)', '中性', '看涨偏斜(负Skew)'],
            '天数': [
                len(bearish_dates),
                len(neutral_dates),
                len(bullish_dates)
            ]
        })
        
        # 创建Skew详细日期表格
        skew_dates_df = pd.concat([
            pd.DataFrame({
                '日期': [format_date(d) for d in bearish_dates],
                'Skew类型': '看跌偏斜(正Skew)',
                'Skew值': [df.loc[d, 'IV_Skew'] for d in bearish_dates]
            }),
            pd.DataFrame({
                '日期': [format_date(d) for d in neutral_dates],
                'Skew类型': '中性',
                'Skew值': [df.loc[d, 'IV_Skew'] for d in neutral_dates]
            }),
            pd.DataFrame({
                '日期': [format_date(d) for d in bullish_dates],
                'Skew类型': '看涨偏斜(负Skew)',
                'Skew值': [df.loc[d, 'IV_Skew'] for d in bullish_dates]
            })
        ], ignore_index=True)
        
        # 添加技术指标说明
        st.write("### 技术指标说明")
        
        # IV水平说明
        st.write("#### 隐含波动率(IV)水平判断标准")
        iv_explain = pd.DataFrame({
            '信号类型': ['高IV', '正常IV', '低IV'],
            '判断标准': [
                'IV > 90%分位数',
                '50%分位数 < IV ≤ 90%分位数',
                'IV ≤ 50%分位数'
            ],
            '市场含义': [
                '市场恐慌情绪较重，预期波动较大，期权价格偏贵',
                '市场情绪平稳，预期波动正常',
                '市场平稳，预期波动较小，期权价格相对便宜'
            ],
            '交易启示': [
                '可考虑做空波动率（卖出期权）策略',
                '可根据其他指标进行综合判断',
                '可考虑做多波动率（买入期权）策略'
            ]
        })
        st.dataframe(iv_explain, use_container_width=True)
        
        # IV Skew说明
        st.write("#### IV Skew（波动率偏斜）判断标准")
        skew_explain = pd.DataFrame({
            'Skew类型': ['看跌偏斜(正Skew)', '中性', '看涨偏斜(负Skew)'],
            '判断标准': [
                'Put IV > Call IV',
                'Put IV ≈ Call IV',
                'Put IV < Call IV'
            ],
            '市场含义': [
                '市场对下跌风险的担忧更大，愿意为认沽期权支付更高的风险溢价',
                '市场对涨跌风险的预期较为平衡',
                '市场对上涨行情的预期更强，愿意为认购期权支付更高的风险溢价'
            ],
            '交易启示': [
                '市场可能处于恐慌状态，可关注超跌反弹机会',
                '可根据其他指标进行综合判断',
                '市场可能处于乐观状态，需警惕过度乐观风险'
            ]
        })
        st.dataframe(skew_explain, use_container_width=True)
        
        # 组合分析说明
        st.write("#### 信号组合分析")
        st.write("""
        1. 最强看跌信号组合：
           - 高IV + 正Skew：表明市场恐慌情绪强烈，预期大幅下跌
           - 交易思路：等待恐慌情绪见顶，布局反弹机会
        
        2. 最强看涨信号组合：
           - 低IV + 负Skew：表明市场情绪平稳偏乐观，预期上涨
           - 交易思路：可考虑低成本做多策略
        
        3. 特殊组合：
           - 高IV + 负Skew：市场剧烈波动，但看涨预期强烈
           - 低IV + 正Skew：市场平稳，但存在下跌担忧
           - 交易思路：需结合其他技术指标和基本面进行综合判断
        """)
        
        # 使用注意事项
        st.warning("""
        注意事项：
        1. IV和Skew信号仅供参考，不能作为唯一的交易依据
        2. 需要结合市场环境、基本面和其他技术指标综合分析
        3. 历史分位数会随着数据更新而变化
        4. 极端市场环境下的信号可能失效
        """)
        
        # 使用选项卡显示Skew统计结果
        tab3, tab4 = st.tabs(["统计数据", "详细日期"])
        
        with tab3:
            st.dataframe(skew_summary, use_container_width=True)
        
        with tab4:
            # 添加Skew类型筛选
            selected_skew = st.selectbox(
                '选择Skew类型',
                ['全部', '看跌偏斜(正Skew)', '中性', '看涨偏斜(负Skew)']
            )
            
            # 筛选数据
            filtered_skew_df = skew_dates_df.copy()
            if selected_skew != '全部':
                filtered_skew_df = filtered_skew_df[filtered_skew_df['Skew类型'] == selected_skew]
            
            # 显示筛选后的数据
            if not filtered_skew_df.empty:
                st.dataframe(
                    filtered_skew_df.sort_values('日期', ascending=False),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("没有符合条件的数据")
    except Exception as e:
        st.error(f"隐含波动率分析出错: {str(e)}")
        st.error("详细错误信息：")
        import traceback
        st.error(traceback.format_exc())

def option_analysis():
    """期权分析页面主函数"""
    st.title("期权市场分析")
    
    # 修改为多文件上传
    uploaded_files = st.file_uploader(
        "上传数据文件 (Excel或CSV格式)",
        type=['xls', 'xlsx', 'csv'],
        accept_multiple_files=True,
        help="可以同时上传多个Excel或CSV文件"
    )
    
    if uploaded_files:
        # 加载和处理数据
        df = load_and_process_data(uploaded_files)
        
        if df is not None:
            # 添加IV分析
            analyze_iv_data(df)
            
            # 显示图表（如果有数据）
            chart = plot_analysis_chart(df)
            if chart is not None:
                st.plotly_chart(chart)
            
            # 市场情绪分析（仅当有足够的数据时）
            if len(df.columns) >= 2:  # 至少有两列数据时进行分析
                signals = analyze_market_sentiment(df)
                
                # 显示分析结果
                st.subheader("市场分析")
                for signal in signals:
                    if signal['signal'] == "看涨":
                        st.success(f"信号: {signal['signal']} ({signal['strength']})\n原因: {signal['reason']}")
                    else:
                        st.warning(f"信号: {signal['signal']} ({signal['strength']})\n原因: {signal['reason']}")
            
            # 添加极值分析
            analyze_market_extremes(df)
            
            # 添加PC比极值区间分析
            analyze_pc_ratio_extremes(df)
            
            # 添加双重PC比信号分析
            analyze_combined_signals(df)
            
            # 添加指标预测能力分析
            analyze_indicator_effectiveness(df)
            
            # 添加PCR区间分析
            analyze_pcr_ranges(df)
            
            # 显示原始数据
            with st.expander("查看原始数据"):
                # 创建数据副本并按日期倒序排列
                display_df = df.sort_index(ascending=False).copy()
                
                # 计算20日均线和标准差
                display_df['上证指数20日均线'] = df['上证指数'].rolling(window=20).mean()
                display_df['上证指数20日标准差'] = df['上证指数'].rolling(window=20).std()
                
                # 计算上下轨
                display_df['上轨'] = display_df['上证指数20日均线'] + 2 * display_df['上证指数20日标准差']
                display_df['下轨'] = display_df['上证指数20日均线'] - 2 * display_df['上证指数20日标准差']
                
                # 设置样式函数
                def style_extreme_values(df):
                    # 创建一个与传入DataFrame大小相同的空样式DataFrame
                    styles = pd.DataFrame('', index=df.index, columns=df.columns)
                    
                    # 获取原始DataFrame中的极值信息
                    high_mask = display_df['上证指数'] > display_df['上轨']
                    low_mask = display_df['上证指数'] < display_df['下轨']
                    
                    # 只对'上证指数'列应用样式
                    if '上证指数' in df.columns:
                        styles.loc[high_mask, '上证指数'] = 'background-color: rgba(255,0,0,0.1); color: red'
                        styles.loc[low_mask, '上证指数'] = 'background-color: rgba(0,255,0,0.1); color: green'
                    
                    return styles
                
                # 添加指标说明
                st.write("数据说明：")
                st.write("- 红色: 上证指数 > 20日均线 + 2倍标准差")
                st.write("- 绿色: 上证指数 < 20日均线 - 2倍标准差")
                
                # 显示统计信息
                stats_df = pd.DataFrame({
                    '指标': ['突破上轨次数', '突破下轨次数'],
                    '次数': [
                        (display_df['上证指数'] > display_df['上轨']).sum(),
                        (display_df['上证指数'] < display_df['下轨']).sum()
                    ]
                })
                st.write("统计信息：")
                st.dataframe(stats_df, use_container_width=True)
                
                # 仅显示原始数据列
                display_columns = [col for col in display_df.columns if not col.endswith(('均线', '标准差', '上轨', '下轨'))]
                final_display_df = display_df[display_columns]
                
                # 应用样式
                styled_df = final_display_df.style.apply(style_extreme_values, axis=None)
                
                # 设置数值格式
                format_dict = {
                    '上证指数': '{:.2f}',
                    '上证指数成交量': '{:.0f}',
                    '华夏上证50ETF期权_认沽认购持仓比': '{:.2f}',
                    '沪深300期权_认沽认购持仓比': '{:.2f}'
                }
                styled_df = styled_df.format(format_dict)
                
                # 显示数据
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=400
                )
                
                # 添加可视化
                fig = go.Figure()
                
                # 添加上证指数
                fig.add_trace(go.Scatter(
                    x=display_df.index,
                    y=display_df['上证指数'],
                    name='上证指数',
                    line=dict(color='blue')
                ))
                
                # 添加均线
                fig.add_trace(go.Scatter(
                    x=display_df.index,
                    y=display_df['上证指数20日均线'],
                    name='20日均线',
                    line=dict(color='gray', dash='dash')
                ))
                
                # 添加上下轨
                fig.add_trace(go.Scatter(
                    x=display_df.index,
                    y=display_df['上轨'],
                    name='上轨(+2σ)',
                    line=dict(color='red', dash='dot')
                ))
                
                fig.add_trace(go.Scatter(
                    x=display_df.index,
                    y=display_df['下轨'],
                    name='下轨(-2σ)',
                    line=dict(color='green', dash='dot')
                ))
                
                fig.update_layout(
                    title='上证指数布林带分析',
                    xaxis_title='日期',
                    yaxis_title='指数',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("请上传数据文件以开始分析")

if __name__ == "__main__":
    option_analysis() 