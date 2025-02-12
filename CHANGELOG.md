# 投资收益计算器更新日志

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

