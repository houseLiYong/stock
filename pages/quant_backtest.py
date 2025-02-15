import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import akshare as ak
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def load_stock_data(stock_code, start_date, end_date):
    """加载股票数据"""
    try:
        # 转换日期格式
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        # 打印调试信息
        st.write(f"正在获取股票数据...")
        st.write(f"股票代码: {stock_code}")
        st.write(f"开始日期: {start_date_str}")
        st.write(f"结束日期: {end_date_str}")
        
        # 移除可能的 sh/sz 前缀
        if stock_code.startswith(('sh', 'sz')):
            stock_code = stock_code[2:]
        
        # 获取数据
        df = ak.stock_zh_a_hist(
            symbol=stock_code, 
            period="daily",
            start_date=start_date_str,
            end_date=end_date_str,
            adjust="qfq"
        )
        
        # 检查数据是否为空
        if df is None or df.empty:
            st.error("获取到的数据为空")
            return None
            
        # 打印数据信息
        st.write("数据获取成功！")
        st.write(f"数据形状: {df.shape}")
        st.write("数据列名:", df.columns.tolist())
        df['日期'] = pd.to_datetime(df['日期'])
         # 将日期转换为数值格式 (例如：20240321)
        df['日期_数值'] = df['日期'].dt.strftime('%Y%m%d').astype(int)
        
        return df
        
    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
        st.write("错误详情:", e)
        return None

def calculate_technical_indicators(df):
    """计算技术指标"""
    # 价格相关
    df['MA5'] = df['收盘'].rolling(window=5).mean()
    df['MA10'] = df['收盘'].rolling(window=10).mean()
    df['MA20'] = df['收盘'].rolling(window=20).mean()
    
    # 成交量相关
    df['VOLUME_MA5'] = df['成交量'].rolling(window=5).mean()
    
    # RSI计算
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    # MACD计算
    def calculate_macd(data, fast=12, slow=26, signal=9):
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    # 计算RSI
    df['RSI'] = calculate_rsi(df['收盘'])
    
    # 计算MACD
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = calculate_macd(df['收盘'])
    
    return df

def prepare_features(df):
    """准备特征数据"""
    features = pd.DataFrame()
    
    try:
        # 价格动量特征
        features['price_ma5'] = df['收盘'] / df['MA5'] - 1
        features['price_ma10'] = df['收盘'] / df['MA10'] - 1
        features['price_ma20'] = df['收盘'] / df['MA20'] - 1
        
        # 成交量特征
        features['volume_ma5'] = df['成交量'] / df['VOLUME_MA5'] - 1
        
        # 技术指标
        features['rsi'] = df['RSI'] / 100  # 归一化RSI
        features['macd'] = df['MACD']
        
        # 添加滞后特征
        for col in features.columns:
            features[f'{col}_lag1'] = features[col].shift(1)
            features[f'{col}_lag2'] = features[col].shift(2)
        
        # 添加变化率
        for col in features.columns:
            if not col.startswith('lag'):
                features[f'{col}_change'] = features[col].pct_change()
        
        # 处理无穷值
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # 处理异常值（将超过3个标准差的值替换为3个标准差的值）
        for col in features.columns:
            mean = features[col].mean()
            std = features[col].std()
            features[col] = features[col].clip(lower=mean-3*std, upper=mean+3*std)
        
        return features
        
    except Exception as e:
        st.error(f"特征生成错误: {str(e)}")
        return None

def calculate_future_returns(df, days=5):
    """计算未来N日收益率"""
    df['future_returns'] = df['收盘'].shift(-days) / df['收盘'] - 1
    return df

class StockRanker:
    """股票评分模型"""
    def __init__(self):
        self.pipeline = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')  # 使用均值填充NaN
    
    def train(self, X, y):
        """训练模型"""
        try:
            # 数据预处理
            st.write("开始数据预处理...")
            
            # 移除y中的NaN值
            valid_mask = ~pd.isna(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            # 检查无穷值
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # 显示数据统计信息
            st.write("数据统计信息：")
            stats = X.describe()
            st.dataframe(stats)
            
            # 创建管道
            self.pipeline = Pipeline([
                ('imputer', self.imputer),  # 首先处理缺失值
                ('scaler', self.scaler),    # 然后进行标准化
                ('regressor', LinearRegression())  # 最后使用线性回归
            ])
            
            # 打印训练数据信息
            st.write("训练数据信息：")
            st.write(f"特征数量: {X.shape[1]}")
            st.write(f"样本数量: {X.shape[0]}")
            st.write("特征名称:", X.columns.tolist())
            
            # 检查并显示缺失值信息
            missing_stats = X.isnull().sum()
            if missing_stats.any():
                st.write("缺失值统计：")
                st.write(missing_stats[missing_stats > 0])
            
            # 训练模型
            st.write("开始训练模型...")
            self.pipeline.fit(X, y)
            
            # 显示训练结果
            train_score = self.pipeline.score(X, y)
            st.success(f"模型训练完成！R² 得分: {train_score:.4f}")
            
            # 显示特征重要性
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': abs(self.pipeline.named_steps['regressor'].coef_)
            }).sort_values('importance', ascending=False)
            
            st.write("特征重要性：")
            st.dataframe(feature_importance)
            
        except Exception as e:
            st.error(f"模型训练错误: {str(e)}")
            raise e
    
    def predict(self, X):
        """预测分数"""
        if self.pipeline is None:
            raise ValueError("模型未训练")
        
        # 处理预测数据中的无穷值
        X = X.replace([np.inf, -np.inf], np.nan)
        
        return self.pipeline.predict(X)

def backtest_strategy(df, positions, initial_capital=1000000):
    """回测策略"""
    portfolio = pd.DataFrame(index=df.index)
    portfolio['holdings'] = positions * df['收盘']
    portfolio['cash'] = initial_capital - portfolio['holdings'].cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    
    return portfolio

def plot_feature_chart(feature_data, feature_name, df_dates):
    """
    绘制特征数据的分布图
    feature_data: 特征数据
    feature_name: 特征名称
    df_dates: 日期数据
    """
    # 创建图表
    fig = go.Figure()
    
    if feature_name == 'rsi':
        # RSI指标的特殊处理
        # 分别添加不同区域的数据
        # RSI > 70 的区域（红色）
        mask_high = feature_data > 70
        fig.add_trace(go.Scatter(
            x=df_dates[mask_high],
            y=feature_data[mask_high],
            mode='lines',
            name='超买区域',
            line=dict(color='red'),
            hovertemplate='日期: %{x}<br>RSI: %{y:.2f}<extra></extra>'
        ))
        
        # RSI < 30 的区域（绿色）
        mask_low = feature_data < 30
        fig.add_trace(go.Scatter(
            x=df_dates[mask_low],
            y=feature_data[mask_low],
            mode='lines',
            name='超卖区域',
            line=dict(color='green'),
            hovertemplate='日期: %{x}<br>RSI: %{y:.2f}<extra></extra>'
        ))
        
        # 30 <= RSI <= 70 的区域（灰色）
        mask_mid = (feature_data >= 30) & (feature_data <= 70)
        fig.add_trace(go.Scatter(
            x=df_dates[mask_mid],
            y=feature_data[mask_mid],
            mode='lines',
            name='正常区域',
            line=dict(color='gray'),
            hovertemplate='日期: %{x}<br>RSI: %{y:.2f}<extra></extra>'
        ))
        
        # 添加参考线
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线(70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线(30)")
        
    else:
        # 其他指标的普通显示
        fig.add_trace(go.Scatter(
            x=df_dates,
            y=feature_data,
            mode='lines',
            name=feature_name,
            hovertemplate='日期: %{x}<br>' + f'{feature_name}: ' + '%{y:.2f}<extra></extra>'
        ))
    
    # 设置图表布局
    fig.update_layout(
        title=f'{feature_name} 走势图',
        xaxis_title='日期',
        yaxis_title=feature_name,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def main():
    st.title("📊 量化回测系统")
    
    # 侧边栏参数设置
    with st.sidebar:
        st.header("参数设置")
        stock_code = st.text_input("股票代码", "000001")
        
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
        end_date = st.date_input("结束日期", datetime.now())
        
        st.markdown("---")
        
        st.subheader("回测参数")
        initial_capital = st.number_input("初始资金（元）", 
                                        min_value=10000, 
                                        max_value=10000000, 
                                        value=100000, 
                                        step=10000,
                                        help="设置回测的初始资金金额")
        
        position_size = st.slider("单次持仓比例", 
                                min_value=0.1, 
                                max_value=1.0, 
                                value=0.5, 
                                step=0.1,
                                help="每次交易使用资金的比例")
        
        st.session_state['backtest_params'] = {
            'initial_capital': initial_capital,
            'position_size': position_size
        }
    
    # 主界面
    tabs = st.tabs(["数据预处理", "特征工程", "模型训练", "回测结果"])
    
    with tabs[0]:
        st.header("数据预处理")
        if st.button("加载数据"):
            with st.spinner("正在加载数据..."):
                df = load_stock_data(stock_code, start_date, end_date)
                if df is not None:
                    st.session_state['raw_data'] = df
                    st.success("数据加载成功！")
                    
                    # 显示数据预览
                    st.subheader("数据预览")
                    st.dataframe(df.head())
                    
                    # 显示数据统计信息
                    st.subheader("数据统计")
                    st.write(f"数据时间范围: {df['日期'].min()} 至 {df['日期'].max()}")
                    st.write(f"总交易天数: {len(df)}")
                    
                    # 显示收盘价走势图
                    st.subheader("收盘价走势")
                    st.line_chart(df.set_index('日期')['收盘'])
    
    with tabs[1]:
        st.header("特征工程")
        if 'raw_data' in st.session_state and st.button("生成特征"):
            df = calculate_technical_indicators(st.session_state['raw_data'])
            features = prepare_features(df)
            st.session_state['features'] = features
            
            # 显示特征数据预览
            st.subheader("特征数据预览")
            st.dataframe(features.head())
            
            # 显示特征解释
            st.subheader("特征指标说明")
            feature_descriptions = {
                'price_ma5': {
                    '名称': '5日均线偏离度',
                    '计算方法': '(当前价格 / 5日均线 - 1) × 100%',
                    '含义': '反映当前价格相对5日均线的偏离程度，正值表示当前价格高于均线，负值表示低于均线',
                    '使用场景': '用于判断短期价格趋势和超买超卖状态'
                },
                'price_ma10': {
                    '名称': '10日均线偏离度',
                    '计算方法': '(当前价格 / 10日均线 - 1) × 100%',
                    '含义': '反映当前价格相对10日均线的偏离程度',
                    '使用场景': '用于判断中短期价格趋势'
                },
                'price_ma20': {
                    '名称': '20日均线偏离度',
                    '计算方法': '(当前价格 / 20日均线 - 1) × 100%',
                    '含义': '反映当前价格相对20日均线的偏离程度',
                    '使用场景': '用于判断中期价格趋势'
                },
                'volume_ma5': {
                    '名称': '5日成交量偏离度',
                    '计算方法': '(当日成交量 / 5日成交量均线 - 1) × 100%',
                    '含义': '反映当前成交量相对近期平均水平的偏离程度',
                    '使用场景': '用于判断成交量异常和市场活跃度'
                },
                'rsi': {
                    '名称': '相对强弱指标(RSI)',
                    '计算方法': '基于14日价格变动计算，范围0-100',
                    '含义': '衡量价格变动的强弱程度',
                    '使用场景': '''
                    - RSI > 70: 可能超买
                    - RSI < 30: 可能超卖
                    - RSI趋势变化: 可能预示价格趋势改变
                    '''
                },
                'macd': {
                    '名称': '指数平滑异同移动平均线(MACD)',
                    '计算方法': '基于12日和26日指数移动平均线',
                    '含义': '反映价格趋势的变化和动能',
                    '使用场景': '''
                    - MACD金叉（由负转正）：可能上涨信号
                    - MACD死叉（由正转负）：可能下跌信号
                    - MACD柱状图：反映趋势强度
                    '''
                }
            }
            
            # 使用expander显示每个特征的详细说明
            for feature, desc in feature_descriptions.items():
                with st.expander(f"📊 {desc['名称']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**计算方法:**")
                        st.write(desc['计算方法'])
                        st.write("**含义:**")
                        st.write(desc['含义'])
                    with col2:
                        st.write("**使用场景:**")
                        st.write(desc['使用场景'])
                        
                    # 如果有数据，显示该特征的分布图
                    if feature in features.columns:
                        st.write("**数据分布:**")
                        # 获取日期数据
                        dates = st.session_state['raw_data']['日期']
                        # 创建并显示图表
                        fig = plot_feature_chart(features[feature], desc['名称'], dates)
                        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.header("模型训练")
        if 'features' in st.session_state and st.button("训练模型"):
            with st.spinner("正在训练模型..."):
                df = calculate_future_returns(st.session_state['raw_data'])
                features = st.session_state['features']
                
                # 去除含有NaN的行
                valid_idx = ~df['future_returns'].isna()
                X = features[valid_idx]
                y = df['future_returns'][valid_idx]
                
                # 显示特征相关性
                st.subheader("特征相关性分析")
                corr_matrix = X.corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu'
                ))
                fig.update_layout(
                    title="特征相关性热力图",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 训练模型
                model = StockRanker()
                model.train(X, y)
                st.session_state['model'] = model
    
    with tabs[3]:
        st.header("回测结果")
        if 'model' in st.session_state and st.button("执行回测"):
            with st.spinner("正在执行回测..."):
                model = st.session_state['model']
                features = st.session_state['features']
                raw_data = st.session_state['raw_data']
                backtest_params = st.session_state['backtest_params']
                
                # 生成预测分数
                scores = model.predict(features)
                
                # 生成交易信号
                positions = np.zeros(len(scores))
                positions[scores > np.percentile(scores, 80)] = 1  # 买入排名前20%的股票
                
                # 执行回测
                portfolio = backtest_strategy(
                    raw_data,
                    positions,
                    backtest_params['initial_capital']
                )
                
                # 显示回测结果
                st.subheader("收益曲线")
                st.line_chart(portfolio['total'])
                
                # 计算回测指标
                total_return = (portfolio['total'].iloc[-1] / backtest_params['initial_capital'] - 1) * 100
                annual_return = total_return * 365 / (end_date - start_date).days
                sharpe_ratio = np.sqrt(252) * portfolio['returns'].mean() / portfolio['returns'].std()
                max_drawdown = ((portfolio['total'] / portfolio['total'].cummax()) - 1).min() * 100
                
                # 显示回测指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总收益率", f"{total_return:.2f}%")
                with col2:
                    st.metric("年化收益率", f"{annual_return:.2f}%")
                with col3:
                    st.metric("夏普比率", f"{sharpe_ratio:.2f}")
                with col4:
                    st.metric("最大回撤", f"{max_drawdown:.2f}%")
                
                # 显示交易统计
                st.subheader("交易统计")
                trade_count = np.sum(np.diff(positions) != 0)
                win_rate = np.sum(portfolio['returns'] > 0) / len(portfolio['returns']) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("交易次数", trade_count)
                with col2:
                    st.metric("胜率", f"{win_rate:.2f}%")

if __name__ == "__main__":
    main() 