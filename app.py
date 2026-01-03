import streamlit as st
import pandas as pd # 注意：这里不需要 import yfinance 了
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tushare as ts # 必须确保安装了 pip install tushare
import time

# 导入 OpenAI 库
try:
    from openai import OpenAI
except ImportError:
    st.error("请先安装 openai 库: pip install openai")

# ==========================================
# 0. 全局配置 (已填入你的真实Key)
# ==========================================
# 🔴 Tushare 代理配置
PROXY_TOKEN = "4987177308688210828" 
PROXY_URL = "http://5k1a.xiximiao.com/dataapi"

# 🔴 阿里云 Key
MY_API_KEY = "sk-ab56f3ab5c694381bec100b8502f99cc" 
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="QuantLens A股透视",
    page_icon="🇨🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("🇨🇳 QuantLens")
    st.info("数据源：Tushare (代理通道)")
    
    st.markdown("---")
    st.header("⚙️ 参数设置")
    ticker_input = st.text_input("输入A股代码 (如 600519)", value="600519")
    
    start_date = st.date_input("开始日期", value=datetime.now() - timedelta(days=365))
    end_date = st.date_input("结束日期", value=datetime.now())
    ma_short = st.number_input("短期均线", value=5, min_value=1)
    ma_long = st.number_input("长期均线", value=20, min_value=1)

# ==========================================
# 2. 数据获取 (🔴 核心修改：使用你的代理Token)
# ==========================================
def generate_mock_data(start_date, end_date):
    """兜底模拟数据"""
    if isinstance(start_date, str): start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str): end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(date_range)
    np.random.seed(42)
    price = 150.0
    prices = []
    for _ in range(n):
        price += np.random.normal(0, 3)
        prices.append(max(price, 10))
    df = pd.DataFrame(index=date_range)
    df['Close'] = prices
    df['Open'] = df['Close'] + np.random.normal(0, 2, n)
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 3, n)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 3, n)
    df['Volume'] = np.random.randint(1000000, 5000000, n)
    return df

@st.cache_data(ttl=3600)
def get_data(code, start, end):
    try:
        # --- 1. 特殊处理：比特币 (BTC) [已修复 Yahoo 报错] ---
        if "BTC" in code.upper():
            import yfinance as yf
            # ❌ 删掉了之前手动创建 session 的代码
            # ✅ 直接调用，让 yfinance 自己处理连接
            tk = yf.Ticker("BTC-USD")
            
            df = tk.history(start=start, end=end)
            
            if df is None or df.empty:
                raise ValueError("Yahoo接口无响应")
            
            # 去除时区
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            return df, False, "BTC-USD"
        # 1. 初始化 Tushare
        pro = ts.pro_api('init_token') # 这里的token随便填，反正下面会改
        
        # 2. 【注入你的代理配置】
        pro._DataApi__token = PROXY_TOKEN
        pro._DataApi__http_url = PROXY_URL
        
        # 3. 处理代码后缀
        code = code.strip()
        if code.isdigit():
            if code.startswith('6'): ts_code = f"{code}.SH"
            elif code.startswith('0') or code.startswith('3'): ts_code = f"{code}.SZ"
            elif code.startswith('8') or code.startswith('4'): ts_code = f"{code}.BJ"
            else: ts_code = f"{code}.SH"
        else:
            ts_code = code

        # 4. 格式化日期 (Tushare要求 YYYYMMDD)
        s_str = start.strftime('%Y%m%d')
        e_str = end.strftime('%Y%m%d')
        
        # 5. 发起请求
        df = pro.daily(ts_code=ts_code, start_date=s_str, end_date=e_str)
        
        if df is None or df.empty:
            raise ValueError(f"接口返回空数据，请检查代码 {ts_code} 是否正确或Token状态")

        # 6. 数据清洗 (适配后续的计算逻辑)
        # Tushare 返回的是倒序的，我们要转成正序
        df = df.sort_values('trade_date')
        df.index = pd.to_datetime(df['trade_date'])
        
        # 重命名列：tushare是小写，程序里用的大写
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'vol': 'Volume'
        })
        
        return df, False, ts_code
        
    except Exception as e:
        # 如果出错了，返回错误信息，而不是模拟数据，方便你看到哪里错了
        return None, True, str(e)

# ==========================================
# 3. 指标计算
# ==========================================
def calculate_indicators(df, ma_s, ma_l):
    data = df.copy()
    data[f'MA_{ma_s}'] = data['Close'].rolling(window=ma_s).mean()
    data[f'MA_{ma_l}'] = data['Close'].rolling(window=ma_l).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['BB_Std']
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['BB_Std']
    data['Signal'] = 0
    data.loc[(data[f'MA_{ma_s}'] > data[f'MA_{ma_l}']) & (data[f'MA_{ma_s}'].shift(1) <= data[f'MA_{ma_l}'].shift(1)), 'Signal'] = 1
    data.loc[(data[f'MA_{ma_s}'] < data[f'MA_{ma_l}']) & (data[f'MA_{ma_s}'].shift(1) >= data[f'MA_{ma_l}'].shift(1)), 'Signal'] = -1
    return data

# ==========================================
# 4. LSTM 预测模块
# ==========================================
def run_lstm_prediction(df, look_back=30, forecast_days=3):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
    except ImportError:
        st.warning("⚠️ 请安装 tensorflow")
        return None, None

    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    
    future_preds = []
    curr_batch = scaled_data[-look_back:].reshape(1, look_back, 1)
    for _ in range(forecast_days):
        pred = model.predict(curr_batch, verbose=0)[0]
        future_preds.append(pred)
        curr_batch = np.append(curr_batch[:, 1:, :], [[pred]], axis=1)
        
    return ([df.index[-1] + timedelta(days=i) for i in range(1, forecast_days+1)], 
            scaler.inverse_transform(future_preds))

# ==========================================
# 5. 通义千问分析函数
# ==========================================
def get_llm_analysis(df, ticker):
    if not MY_API_KEY or "sk-" not in MY_API_KEY:
        return "⚠️ 代码配置错误：请在代码第18行填入正确的阿里云 API Key (sk-开头)。"
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    ma_s_val = last[f'MA_{ma_short}']
    ma_l_val = last[f'MA_{ma_long}']
    
    prompt = f"""
    作为A股资深分析师，请根据 {ticker} 的最新数据写简报（Markdown，200字内）：
    1. 最新价: {last['Close']:.2f} (日涨跌: {((last['Close']-prev['Close'])/prev['Close'])*100:.2f}%)
    2. RSI(14): {last['RSI']:.2f}
    3. 均线: MA{ma_short}={ma_s_val:.2f}, MA{ma_long}={ma_l_val:.2f} ({"金叉" if ma_s_val > ma_l_val else "死叉"})
    4. 布林带: 上轨{last['BB_Upper']:.2f}, 下轨{last['BB_Lower']:.2f}
    请结合A股市场风格，给出操作建议。
    """
    
    try:
        client = OpenAI(api_key=MY_API_KEY, base_url=BASE_URL)
        response = client.chat.completions.create(
            model="qwen-plus", 
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Qwen 调用失败 (请检查Key是否有效): {str(e)}"

# ==========================================
# 6. 主界面
# ==========================================
st.title(f"📊 QuantLens A股: {ticker_input} 分析")

# 获取数据，并拿到自动修正后的代码（比如 600519 -> 600519.SH）
df_raw, is_error, msg = get_data(ticker_input, start_date, end_date)

if is_error:
    st.error(f"❌ 数据获取失败！", icon="🚨")
    st.error(f"错误信息: {msg}")
    st.warning("请检查：Token是否正确？或者尝试输入完整的代码如 600519.SH")
    # 出错时显示模拟数据，方便演示布局
    df_raw = generate_mock_data(start_date, end_date)
    ts_code_display = "模拟数据(Mock)"
else:
    ts_code_display = msg # 第三个返回值是代码

if df_raw is not None:
    df = calculate_indicators(df_raw, ma_short, ma_long)
    last = df.iloc[-1]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("收盘价", f"{last['Close']:.2f}", f"{((last['Close']-df.iloc[-2]['Close'])/df.iloc[-2]['Close'])*100:.2f}%")
    c2.metric("RSI", f"{last['RSI']:.2f}")
    c3.metric("信号", "金叉/看多" if df['Signal'].iloc[-1]==1 else ("死叉/看空" if df['Signal'].iloc[-1]==-1 else "观望"))
    c4.metric("布林位置", "超买" if last['Close']>last['BB_Upper'] else "正常")

    tab1, tab2 = st.tabs(["📊 技术视图", "📄 原始数据"])
    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.2, 0.7])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K线'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f'MA_{ma_short}'], line=dict(color='orange'), name=f'MA{ma_short}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f'MA_{ma_long}'], line=dict(color='blue'), name=f'MA{ma_long}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', dash='dot'), name='Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], fill='tonexty', line=dict(color='gray', dash='dot'), name='Lower'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
        
        st.plotly_chart(fig, use_container_width=True) 

    with tab2: st.dataframe(df.sort_index(ascending=False))

    st.markdown("---")
    c_lstm, c_llm = st.columns(2)
    
    with c_lstm:
        st.subheader("📈 LSTM 趋势预测")
        if st.button("运行神经网络预测"):
            with st.spinner("计算中..."):
                d, p = run_lstm_prediction(df)
            if d:
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(x=df.index[-30:], y=df['Close'][-30:], name='历史', line=dict(color='cyan')))
                fig_p.add_trace(go.Scatter(x=d, y=p.flatten(), name='预测', line=dict(color='yellow', dash='dot')))
                fig_p.update_layout(height=300, template="plotly_dark", margin=dict(t=10,b=10,l=10,r=10))
                st.plotly_chart(fig_p, use_container_width=True)
                st.info(f"未来3天趋势预测：{'看涨 📈' if p[-1]>p[0] else '看跌 📉'}")

    with c_llm:
        st.subheader("🤖 通义千问 A股研报")
        if st.button("生成分析报告"):
            with st.spinner("Qwen-Plus 正在分析A股行情..."):
                rep = get_llm_analysis(df, ts_code_display)
                st.markdown(f"<div style='background:#f0f2f6;color:black;padding:15px;border-radius:10px;'>{rep}</div>", unsafe_allow_html=True)
    # ==========================================
    # 7. 新增：AI 智能投资顾问 (放在最末尾)
    # ==========================================
    st.markdown("---")
    st.subheader("💬 AI 投资顾问")
    st.caption("您可以问：'现在适合买入吗？' 或 '帮我分析一下支撑位'")

    # 1. 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. 处理用户输入
    if prompt := st.chat_input("关于这只股票，你还想问什么？"):
        # 显示用户的问题
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 构建给 AI 的背景知识 (System Context)
        # 把刚才算出来的指标喂给 AI，这样它才知道你在问哪只股票
        context_data = f"""
        你是一个专业的A股投资顾问。
        当前讨论的股票代码：{ticker_input}。
        最新技术指标数据（截止{last.name.strftime('%Y-%m-%d')}）：
        - 收盘价：{last['Close']:.2f}
        - RSI(14)：{last['RSI']:.2f}
        - 均线状态：短期MA{ma_short}={last[f'MA_{ma_short}']:.2f}, 长期MA{ma_long}={last[f'MA_{ma_long}']:.2f}
        - 布林带：上轨{last['BB_Upper']:.2f}, 下轨{last['BB_Lower']:.2f}
        
        用户的问题是：{prompt}
        请基于以上数据简要回答。
        """

        # 调用通义千问 API
        with st.chat_message("assistant"):
            try:
                client = OpenAI(api_key=MY_API_KEY, base_url=BASE_URL)
                stream = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": context_data}, # 把背景知识埋在系统提示里
                        {"role": "user", "content": prompt}
                    ],
                    stream=True, # 开启流式输出，像打字机一样
                )
                response = st.write_stream(stream) # 实时显示回答
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            except Exception as e:

                st.error(f"AI 响应失败: {e}")

