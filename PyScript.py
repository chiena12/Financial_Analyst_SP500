import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.utils import class_weight
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from lazypredict.Supervised import LazyClassifier

# Cấu hình trang Streamlit
st.set_page_config(page_title="S&P 500 Stock Analysis", layout="wide")
st.title("Phân tích và Dự báo Danh mục Cổ phiếu S&P 500")

# -------------------------------------------------------------------
# 1. DATA PREPARING (Sử dụng cache để không phải tải lại liên tục)
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("innacampo/s-and-p-500-stocks-daily-historical-data-10-years")
    subdir = f"{path}/SP500_Data_10Y"
    
    selected_files = ['ACGL.csv', 'FIS.csv', 'FITB.csv', 'IEX.csv', 'LOW.csv']
    list_file = []
    column_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    for file_name in selected_files:
        file_path = os.path.join(subdir, file_name)
        df = pd.read_csv(file_path, skiprows=3, names=column_names, header=None)
        df['ticker'] = file_name.replace('.csv', '')
        list_file.append(df)
        
    data = pd.concat(list_file, ignore_index=True)
    data['Date'] = pd.to_datetime(data['Date'])
    
    portfolio_data = data.pivot(index='Date', columns='ticker', values='Close')
    return data, portfolio_data

with st.spinner("Đang tải dữ liệu..."):
    data, portfolio_data = load_data()

st.header("EDA (Exploratory Data Analysis)")
st.subheader("TỔNG QUAN DỮ LIỆU")

# Bảng portfolio_data.describe() CÓ NHẬN XÉT -> IN RA
st.dataframe(portfolio_data.describe())
st.markdown("""
**Nhận xét:**
1. Mã LOW có sự tăng trưởng mạnh mẽ nhất khi tăng từ đáy 53 USD lên đỉnh 277 USD
2. Mã IEX và LOW có độ lệch chuẩn (std) cao chứng tỏ sự biến động mạnh của 2 mã này đi kèm với rủi ro biến động cao
3. Giá trung bình của FITB là thấp nhất ; IEX và LOW là cao nhất
""")

# -------------------------------------------------------------------
# 2. VẼ BIỂU ĐỒ LỢI NHUẬN TÍCH LUỸ
# -------------------------------------------------------------------
st.subheader("VẼ BIỂU ĐỒ LỢI NHUẬN TÍCH LUỸ")
fig1, ax1 = plt.subplots(figsize=(10, 6))
cumulative_returns = portfolio_data / portfolio_data.iloc[0]
for column in cumulative_returns.columns:
    ax1.plot(cumulative_returns.index, cumulative_returns[column], label=column, linewidth=1)
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Returns')
ax1.set_title('Cumulative Returns of All Stocks')
ax1.legend()
st.pyplot(fig1)

st.markdown("""
**Nhận xét:**
1. ACGL bứt phá mạnh mẽ từ sau 2021 với lợi nhuận tích luỹ đã tăng gấp 4,5 lần vào thời điểm này và đỉnh cao là gấp 5 lần vào năm 2024 . LOW cũng có mức tăng trưởng ấn tượng với con số từ 3,5 đến 4,3 lần.
2. ITEX có mức tăng trưởng bền vững với con số từ 2,5 đến 3,5 lần ; với FITB là từ 2 đến 3 lần
3. FIS là mã có ít sự bứt phá nhất khi lợi nhuận tích luỹ hiện tại chỉ khoảng 1,2 lần và cao nhất cũng chỉ là 2,5 lần
""")

# -------------------------------------------------------------------
# 3. VẼ BIỂU ĐỒ PHÂN TÍCH GIÁ TRỊ NGOẠI LAI (OUTLIERS)
# -------------------------------------------------------------------
st.subheader("VẼ BIỂU ĐỒ PHÂN TÍCH GIÁ TRỊ NGOẠI LAI (OUTLIERS)")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=portfolio_data, ax=ax2)
ax2.set_title('Outliers Analysis')
ax2.set_ylabel('Stocks Price')
st.pyplot(fig2)

st.markdown("""
**Nhận xét:**
1. Dữ liệu không có giá trị ngoại lai (Outliers) => Không có các phiên giá tăng hay giảm về mức vô lý ( Dữ liệu không bị sai và lỗi)
2. Biến động nằm trong tầm kiểm soát và không phải do các biến động tức thời
""")

# -------------------------------------------------------------------
# 4. VẼ BIỂU ĐỒ HIỆP PHƯƠNG SAI VÀ ĐỘ TƯƠNG QUAN
# -------------------------------------------------------------------
st.subheader("VẼ BIỂU ĐỒ HIỆP PHƯƠNG SAI VÀ ĐỘ TƯƠNG QUAN GIỮA CÁC CỔ PHIẾU")
cov_matrix = portfolio_data.pct_change().dropna().cov()
corr_matrix = portfolio_data.pct_change().dropna().corr()

fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt='.6f', ax=ax3[0])
ax3[0].set_title('Covariance Matrix')

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax3[1])
ax3[1].set_title('Correlation Matrix')

plt.tight_layout()
st.pyplot(fig3)

st.markdown("""
**Nhận xét:**
1. Các hệ số tương quan nằm trong khoảng từ 0.36 đến 0.56 => Đây là mức tương quan trung bình và rất phù hợp để đa dạng hoá danh mục đầu tư , các cổ phiếu không quá phụ thuộc nhau vào ngắn hạn
2. ACGL và LOW có hệ số tương quan là 0.36 dựa trên lợi nhuận hàng ngày cho thấy mô hình Makorwitz có thể phân bổ đáng kể vào cặp đôi này để tối ưu Sharpe Ratio
""")

# -------------------------------------------------------------------
# 5. FEATURE ENGINEERING & SHARPE RATIO TABLE
# -------------------------------------------------------------------
st.header("FEATURE ENGINEERING")
daily_returns = portfolio_data.pct_change().dropna()
daily_mean = daily_returns.mean()
daily_std = daily_returns.std()

annual_mean = daily_mean * 252
annual_volatility = daily_std * np.sqrt(252)
sharpe_ratio = annual_mean / annual_volatility

comparison_table = pd.DataFrame({
    'Annual Mean': annual_mean,
    'Annual Volatility': annual_volatility,
    'Sharpe Ratio': sharpe_ratio
})
comparison_table = comparison_table.sort_values(by='Sharpe Ratio', ascending=False)

# Bảng comparison_table CÓ NHẬN XÉT -> IN RA
st.dataframe(comparison_table)
st.markdown("""
**Nhận xét:**
1. ACGL có Sharpe Ratio cao nhất và là cổ có hiệu quả tốt nhất danh mục
2. FITB tuy có lợi nhuận trung bình hằng năm cao nhất nhưng do biến động quá cao nên làm Sharpe Ratio giảm đáng kể
3. IEX có lợi nhuận mức trung bình 12,78% nhưng lại có độ biến động thấp nhất trong 5 cổ phiếu và sẽ là sự lựa chọn an toàn nhất trong rổ khi thị trường biến động
""")

# -------------------------------------------------------------------
# 6. MONTE CARLO SIMULATION
# -------------------------------------------------------------------
st.header("MODELLING")
st.subheader("MÔ PHỎNG 20000 KỊCH BẢN CHIA DANH MỤC KHÁC NHAU (MONTE CARLO)")

exp_returns = annual_mean
cov_matrix_mc = daily_returns.cov() * 252
num_assets = len(exp_returns)
num_simulations = 20000

# Cache Monte Carlo để tránh chạy lại tốn thời gian khi tương tác
@st.cache_data
def run_monte_carlo(exp_returns, cov_matrix_mc, num_assets, num_simulations):
    results = np.zeros((num_assets+3, num_simulations))
    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(exp_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_mc, weights)))
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        results[2,i] = results[0,i] / results[1,i]
        for j in range(num_assets):
            results[j+3,i] = weights[j]
    
    columns = ['Returns', 'Volatility', 'Sharpe Ratio'] + [ticker for ticker in exp_returns.index]
    return pd.DataFrame(results.T, columns=columns)

results_df = run_monte_carlo(exp_returns, cov_matrix_mc, num_assets, num_simulations)

max_sharpe_ratio = results_df.iloc[results_df['Sharpe Ratio'].idxmax()]
min_vol_ratio = results_df.iloc[results_df['Volatility'].idxmin()]

st.subheader("VẼ ĐƯỜNG BIÊN HIỆU QUẢ")
fig4, ax4 = plt.subplots(figsize=(12, 8))
scatter = ax4.scatter(results_df.Volatility, results_df.Returns, c=results_df['Sharpe Ratio'], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Sharpe Ratio')
ax4.scatter(max_sharpe_ratio.Volatility, max_sharpe_ratio.Returns, color='r', marker='*', s=200, label='Max Sharpe Ratio')
ax4.scatter(min_vol_ratio.Volatility, min_vol_ratio.Returns, color='b', marker='*', s=200, label='Min Volatility')
ax4.set_title('Danh mục hiệu quả - Giả lập Monte Carlo')
ax4.set_xlabel('Biến động theo năm')
ax4.set_ylabel('Lợi nhuận trung bình hằng năm')
ax4.legend()
st.pyplot(fig4)

st.markdown("""
**Nhận xét:**
1. Ngôi sao đỏ chính là vùng đầu tư thông minh nhất dựa trên chỉ số Sharpe Ratio đem lại mức bù đắp lợi nhuận lớn nhất dựa trên mỗi đơn vị rủi ro (18%)
2. Ngôi sao xanh chính là vùng an toàn nhất với độ biến động thấp nhất nhưng bù lại lợi nhuận đem lại chỉ ở mức 14%
3. Đường cánh cung được nối từ sao đỏ với sao xanh chính là vùng đầu tư đối với rổ cố phiếu này khi với mỗi điểm ở cánh cung này thì không thể có điểm có lợi nhuận cao hơn mà rủi ro không tăng
""")

# -------------------------------------------------------------------
# 7. CHUẨN BỊ DỮ LIỆU CHO DEEP LEARNING (ACGL)
# -------------------------------------------------------------------
st.header("SO SÁNH GIỮA VIỆC SỬ DỤNG BI-LSTM VÀ XGBOOST ĐỂ DỰ ĐOÁN XU HƯỚNG TĂNG/GIẢM VÀO NGÀY TIẾP THEO (ACGL)")

data_acgl = data[data['ticker'] == 'ACGL'].copy()
daily_return_acgl = data_acgl.Close.pct_change()
data_acgl['Log_Daily_Return'] = np.log(1 + daily_return_acgl)
data_acgl['Return'] = daily_return_acgl
data_acgl['Lag_Log_Return'] = data_acgl['Log_Daily_Return'].shift(1)

data_acgl['Volume_Change'] = data_acgl.groupby('ticker').Volume.pct_change()
volume_ma_20 = data_acgl.groupby('ticker').Volume.transform(lambda x: x.rolling(20).mean())
data_acgl['Volume_Change_MA20'] = data_acgl.Volume / volume_ma_20

data_acgl['RSI_14'] = data_acgl.groupby('ticker').Close.transform(lambda x: ta.rsi(x, length=14))
data_acgl['RSI_14_MA60'] = data_acgl.groupby('ticker')['RSI_14'].transform(lambda x: x.rolling(60).mean())

data_acgl['EMA_20'] = data_acgl.groupby('ticker').Close.transform(lambda x: x.ewm(span=20, adjust=False).mean())
data_acgl['EMA_Cross'] = data_acgl['EMA_20'] - data_acgl.groupby('ticker').Close.transform(lambda x: x.ewm(span=60, adjust=False).mean())
data_acgl['Dist_EMA_60'] = (data_acgl['Close'] - data_acgl.groupby('ticker').Close.transform(lambda x: x.ewm(span=60, adjust=False).mean())) / data_acgl.groupby('ticker').Close.transform(lambda x: x.ewm(span=60, adjust=False).mean())

data_acgl['Rolling_Std_20'] = data_acgl.groupby('ticker').Close.transform(lambda x: x.rolling(20).std())
data_acgl['H_L_Spread'] = (data_acgl['High'] - data_acgl['Low']) / data_acgl['Close']
data_acgl['Target_Class'] = np.where(data_acgl['Log_Daily_Return'].shift(-1) > 0, 1, 0)
data_acgl.dropna(inplace=True)

# -------------------------------------------------------------------
# 8. MÔ HÌNH HỌC SÂU (BI-LSTM)
# -------------------------------------------------------------------
features = ['Volume_Change', 'Volume_Change_MA20', 'RSI_14', 'RSI_14_MA60', 'EMA_20', 'EMA_Cross', 'Rolling_Std_20', 'H_L_Spread']
final_data = data_acgl.copy()
train_size = int(len(final_data) * 0.8)
train_data = final_data.iloc[:train_size].copy()
test_data = final_data.iloc[train_size:].copy()

scaler_x = MinMaxScaler(feature_range=(0, 1))
train_data[features] = scaler_x.fit_transform(train_data[features])
test_data[features] = scaler_x.transform(test_data[features])

def create_sequences(df, length):
    x, y = [], []
    for ticker in df.ticker.unique():
        ticker_df = df[df['ticker'] == ticker]
        ticker_values = ticker_df[features + ['Target_Class']].values
        for i in range(length, len(ticker_values)):
            x.append(ticker_values[i-length:i, :-1])
            y.append(ticker_values[i, -1])
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_data, 60)
x_test, y_test = create_sequences(test_data, 60)

# Huấn luyện Bi-LSTM
with st.spinner('Đang huấn luyện mô hình Bi-LSTM... (Có thể mất một vài phút)'):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=20, return_sequences=True, kernel_regularizer=regularizers.l2(0.01), input_shape=(x_train.shape[1], x_train.shape[2]))))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(units=32)))
    model.add(Dropout(0.35))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(weights))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Giảm epochs xuống một chút hoặc để nguyên, tuỳ vào resource của Streamlit (ở đây giữ nguyên 50)
    model.fit(x_train, y_train, epochs=50, batch_size=32, callbacks=[early_stopping], validation_split=0.1, verbose=0, shuffle=False, class_weight=class_weights_dict)

y_pred_probs = model.predict(x_test)
y_pred_labels = (y_pred_probs > np.median(y_pred_probs)).astype(int).flatten()

st.subheader("VẼ BIỂU ĐỒ SO SÁNH THỰC TẾ VÀ DỰ ĐOÁN (BI-LSTM)")
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.plot(y_test[:50], label='Thực tế (1: Tăng, 0: Giảm)', marker='o', linestyle='', alpha=0.6)
ax5.plot(y_pred_labels[:50], label='Dự báo', marker='x', linestyle='', color='red')
ax5.set_xlabel('Thời gian')
ax5.set_ylabel('Nhãn')
ax5.set_title('So sánh thực tế và dự báo')
ax5.legend()
st.pyplot(fig5)

st.subheader("KIỂM TRA PHÂN PHỐI GIỮA CÁC NHÃN")
fig6, ax6 = plt.subplots()
sns.histplot(y_pred_probs, kde=True, ax=ax6)
ax6.set_title("Phân phối xác suất ở trong phần dự báo")
st.pyplot(fig6)

# Đoạn code mới: Tính toán và in số lượng nhãn ra Streamlit
counts_lstm = pd.Series(y_pred_labels).value_counts()
col1, col2 = st.columns(2)
col1.metric("Số lượng nhãn 0 (Giảm/Sideway)", counts_lstm.get(0, 0))
col2.metric("Số lượng nhãn 1 (Tăng)", counts_lstm.get(1, 0))

st.subheader("VẼ BIỂU ĐỒ SO SÁNH LỢI NHUẬN GIỮA VIỆC SỬ DỤNG BI-LSTM VÀ CHIẾN THUẬT MUA RỒI GIỮ")
y_test_returns = test_data['Log_Daily_Return'].values[60:]
results = pd.DataFrame({'Actual_Return': y_test_returns.flatten(), 'Prob_Up': y_pred_probs.flatten()})
results['Signal'] = (results['Prob_Up'] > np.median(results['Prob_Up'])).astype(int)
results['Strategy_Return'] = results['Signal'].shift(1) * results['Actual_Return']
results.dropna(inplace=True)

results['Market_Cum'] = results['Actual_Return'].cumsum().apply(np.exp)
results['Strategy_Cum'] = results['Strategy_Return'].cumsum().apply(np.exp)

sharpe = np.sqrt(252) * (results['Strategy_Return'].mean() - 0) / results['Strategy_Return'].std()
peak = results['Strategy_Cum'].cummax()
drawdown = (results['Strategy_Cum'] - peak) / peak
max_drawdown = drawdown.min()

fig7, ax7 = plt.subplots(figsize=(17, 7))
ax7.plot(results['Market_Cum'], label='Thị trường (Mua rồi giữ)', color='gray', alpha=0.5, linestyle='--')
ax7.plot(results['Strategy_Cum'], label='Bi-LSTM Strategy', color='blue', linewidth=2)
ax7.set_title(f'KẾT QUẢ BACKTEST\nSharpe Ratio: {sharpe:.2f} | Max Drawdown: {max_drawdown*100:.2f}%', fontsize=14)
ax7.set_xlabel('Thời gian (Số phiên)')
ax7.set_ylabel('Tỷ lệ sinh lời dựa trên tài sản gốc')
ax7.legend()
ax7.grid(True, alpha=0.3)
st.pyplot(fig7)

st.write(f"**Mức sinh lời của thị trường:** {(results['Market_Cum'].iloc[-1]-1)*100:.2f}%")
st.write(f"**Mức sinh lời của chiến thuật:** {(results['Strategy_Cum'].iloc[-1]-1)*100:.2f}%")


# -------------------------------------------------------------------
# 9. MÔ HÌNH HỌC MÁY (XGBOOSTCLASSIFIER)
# -------------------------------------------------------------------
st.header("TỐI ƯU HÓA MÔ HÌNH XGBOOST (GRIDSEARCHCV)")

# Đảm bảo danh sách features không bị trùng lặp khi re-run
ml_features = [f for f in features if f != 'Lag_Log_Return']
if 'Lag_Log_Return' not in ml_features:
    ml_features.append('Lag_Log_Return')

X_train_ml = train_data[ml_features]
y_train_ml = train_data['Target_Class']
X_test_ml = test_data[ml_features]
y_test_ml = test_data['Target_Class']

# 1. Định nghĩa lưới tham số (Grid) - Giữ ở mức vừa phải để chạy ổn định trên Cloud
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# 2. Sử dụng TimeSeriesSplit thay cho K-Fold thông thường vì đây là dữ liệu chuỗi thời gian
tscv = TimeSeriesSplit(n_splits=3)

with st.spinner('Đang tìm kiếm bộ tham số tối ưu (GridSearchCV)... Vui lòng đợi trong giây lát.'):
    xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)
    
    rand_search = RandomizedSearchCV(
        estimator=xgb_base, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='accuracy', # Tận dụng đa nhân CPU
    )
    rand_search.fit(X_train_ml, y_train_ml)

# 3. Lấy bộ tham số và mô hình tốt nhất
best_params = rand_search.best_params_
best_xgb = rand_search.best_estimator_

# Hiển thị bộ tham số tốt nhất lên giao diện
st.success("Đã tìm thấy bộ tham số tối ưu!")
st.write("**Best Parameters:**")
st.json(best_params)

# Dự báo với mô hình đã tối ưu
y_pred_probs_ml = best_xgb.predict_proba(X_test_ml)[:, 1]
threshold_ml = np.median(y_pred_probs_ml)
y_pred_labels_ml = (y_pred_probs_ml > threshold_ml).astype(int)

# --- PHẦN VẼ BIỂU ĐỒ (Giữ nguyên logic Matplotlib an toàn) ---
st.subheader("VẼ BIỂU ĐỒ PHÂN PHỐI NHÃN (XGBOOST OPTIMIZED)")
fig8, ax8 = plt.subplots(figsize=(8, 4))
sns.histplot(y_pred_probs_ml, kde=True, ax=ax8, color='skyblue')
ax8.set_title("Phân phối xác suất dự báo sau khi tối ưu")
st.pyplot(fig8)
plt.close(fig8)

# In số lượng nhãn
counts_xgb = pd.Series(y_pred_labels_ml).value_counts()
c1, c2 = st.columns(2)
c1.metric("Nhãn 0 (Giảm/Sideway)", counts_xgb.get(0, 0))
c2.metric("Nhãn 1 (Tăng)", counts_xgb.get(1, 0))

# --- FEATURE IMPORTANCE ---
st.subheader("MỨC ĐỘ QUAN TRỌNG CÁC FEATURES (BEST MODEL)")
feature_importance = pd.Series(best_xgb.feature_importances_, index=ml_features)
fig9, ax9 = plt.subplots(figsize=(10, 6))
feature_importance.nlargest(10).plot(kind='barh', color='purple', ax=ax9)
ax9.set_title('Top 10 Features ảnh hưởng đến dự báo')
st.pyplot(fig9)


st.markdown("**CÁC FEATURES CÓ ĐÓNG GÓP GẦN NHƯ TƯƠNG ĐƯƠNG NHAU**")

st.subheader("VẼ BIỂU ĐỒ SO SÁNH LỢI NHUẬN GIỮA VIỆC SỬ DỤNG XGBOOST VÀ CHIẾN THUẬT MUA RỒI GIỮ")
y_test_ml_returns = test_data['Log_Daily_Return'].values[len(test_data) - len(y_pred_probs_ml) :]
res = pd.DataFrame({'Actual_Return': y_test_ml_returns.flatten(), 'Prob_Up': y_pred_probs_ml.flatten()})
res['Signal'] = (res['Prob_Up'] > np.median(res['Prob_Up'])).astype(int)
res['Strategy_Return'] = res['Signal'].shift(1) * res['Actual_Return']
res.dropna(inplace=True)

res['Market_Cum'] = res['Actual_Return'].cumsum().apply(np.exp)
res['Strategy_Cum'] = res['Strategy_Return'].cumsum().apply(np.exp)

sharpe_xgb = np.sqrt(252) * (res['Strategy_Return'].mean() - 0) / res['Strategy_Return'].std()
peak_xgb = res['Strategy_Cum'].cummax()
drawdown_xgb = (res['Strategy_Cum'] - peak_xgb) / peak_xgb
max_drawdown_xgb = drawdown_xgb.min()

fig10, ax10 = plt.subplots(figsize=(17, 7))
ax10.plot(res['Market_Cum'], label='Thị trường (Mua rồi giữ)', color='gray', alpha=0.5, linestyle='--')
ax10.plot(res['Strategy_Cum'], label='XGBOOST', color='blue', linewidth=2)
ax10.set_title(f'KẾT QUẢ BACKTEST\nSharpe Ratio: {sharpe_xgb:.2f} | Max Drawdown: {max_drawdown_xgb*100:.2f}%', fontsize=14)
ax10.set_xlabel('Thời gian (Số phiên)')
ax10.set_ylabel('Tỷ lệ sinh lời dựa trên tài sản gốc')
ax10.legend()
ax10.grid(True, alpha=0.3)
st.pyplot(fig10)

st.write(f"**Mức sinh lời của thị trường:** {(res['Market_Cum'].iloc[-1]-1)*100:.2f}%")
st.write(f"**Mức sinh lời của chiến thuật:** {(res['Strategy_Cum'].iloc[-1]-1)*100:.2f}%")

# -------------------------------------------------------------------
# 10. KẾT LUẬN
# -------------------------------------------------------------------
st.header("KẾT LUẬN")
st.markdown("""
**Mô hình Bi-LSTM:** Nhờ vào kiến trúc mạng nơ-ron hồi quy hai chiều, mô hình có khả năng khai thác tối ưu tính phụ thuộc thời gian (time-dependency) và các mẫu hình phi tuyến tính phức tạp trong dữ liệu lịch sử. Khả năng "ghi nhớ" dài hạn giúp Bi-LSTM bắt kịp các điểm đảo chiều xu hướng một cách khá nhạy bén với minh chứng là accuaracy của tập val lên đến hơn 57%. Tuy nhiên, do quy mô dữ liệu quá nhỏ nên mô hình rất dễ bị overfitting và đây cũng không phải là cách tiếp cận tối ưu nhất.

**Mô hình XGBoost:** Là một thuật toán mạnh mẽ dựa trên Boosting cây quyết định, nhưng đối với dữ liệu chuỗi thời gian tài chính có độ nhiễu cao, XGBoost dễ rơi vào tình trạng quá khớp (overfitting) hoặc phản ứng chậm với các biến động ngắn hạn nếu thiếu các đặc trưng kỹ thuật chuyên sâu. Tuy nhiên, mô hình này vẫn có ứng dụng trong thực tiễn cao.
""")