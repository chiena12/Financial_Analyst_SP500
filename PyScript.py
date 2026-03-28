# %% [markdown]
# ## DATA PREPARING

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("innacampo/s-and-p-500-stocks-daily-historical-data-10-years")

print("Path to dataset files:", path)

# %%
# Directory path : /root/.cache/kagglehub/datasets/innacampo/s-and-p-500-stocks-daily-historical-data-10-years/versions/1
import pandas as pd
import os
directory = os.listdir(path)
print(directory)

# %%
subdir = f"{path}/SP500_Data_10Y"


# Chọn 5 cổ phiếu để đưa vào rổ
selected_files =['ACGL.csv','FIS.csv','FITB.csv','IEX.csv','LOW.csv']
list_file=[]

# Danh sách tên cột
column_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
for file_name in selected_files:

    #Tạo đường dẫn đến từng file
    file_path = os.path.join(subdir, file_name)

    # Đọc file vào DataFrame
    df = pd.read_csv(file_path, skiprows=3, names=column_names, header=None)

    # Thêm cột Ticker để phẩn biệt data thuộc cổ phiếu nào
    df['ticker'] = file_name.replace('.csv','')

    # Thêm file vào danh sách
    list_file.append(df)

# Gộp danh sách các cột
data=pd.concat(list_file,ignore_index=True)

data.head(5)

# %%
data.info()

# %%
# Chuyển kiểu dữ liệu cột Date sang Datetime
data['Date']=pd.to_datetime(data['Date'])
data.info()

# %%
# Chuyển từ wide table sang long table
portfolio_data=data.pivot(index='Date',columns='ticker',values='Close')
portfolio_data.head(5)

# %%
portfolio_data.shape

# %% [markdown]
# => Bảng porfolio_data gồm 5 cột và 2515 dòng

# %%
portfolio_data.isnull().sum()

# %% [markdown]
# => Không có giá trị bị thiếu (No missing values)

# %% [markdown]
# ## EDA (Exploratory Data Analysis)

# %% [markdown]
# # TỔNG QUAN DỮ LIỆU

# %%
portfolio_data.describe()

# %% [markdown]
# Nhận xét :
# 
# 1, Mã LOW có sự tăng trưởng mạnh mẽ nhất khi tăng từ đáy 53 USD lên đỉnh 277 USD
# 
# 2, Mã IEX và LOW có độ lệch chuẩn (std) cao chứng tỏ sự biến động mạnh của 2 mã này đi kèm với rủi ro biến động cao
# 
# 3, Giá trung bình của FITB là thấp nhất ; IEX và LOW là cao nhất

# %% [markdown]
# # VẼ BIỂU ĐỒ LỢI NHUẬN TÍCH LUỸ

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
plt.figure(figsize=(10, 6))

# Tính lợi nhuận tích luỹ = Tất cả các dòng / Dòng đầu tiên
cumulative_returns= portfolio_data/portfolio_data.iloc[0]
for column in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[column], label=column,linewidth=1)
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Cumulative Returns of All Stocks')
plt.legend()
plt.show()



# %% [markdown]
# Nhận xét:
# 
# 1, ACGL bứt phá mạnh mẽ từ sau 2021 với lợi nhuận tích luỹ đã tăng gấp 4,5 lần vào thời điểm này và đỉnh cao là gấp 5 lần vào năm 2024 . LOW cũng có mức tăng trưởng ấn tượng với con số từ 3,5 đến 4,3 lần.
# 
# 2, ITEX có mức tăng trưởng bền vững với con số từ 2,5 đến 3,5 lần ; với FITB là từ 2 đến 3 lần
# 
# 3, FIS là mã có ít sự bứt phá nhất khi lợi nhuận tích luỹ hiện tại chỉ khoảng 1,2 lần và cao nhất cũng chỉ là 2,5 lần
# 
# 

# %% [markdown]
# # VẼ BIỂU ĐỒ PHÂN TÍCH GIÁ TRỊ NGOẠI LAI (OUTLIERS)

# %%
# Phân tích các outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=portfolio_data)
plt.title('Outliers Analysis')
plt.ylabel('Stocks Price')
plt.show()

# %% [markdown]
# Nhận xét:
# 
# 1, Dữ liệu không có giá trị ngoại lai (Outliers) => Không có các phiên giá tăng hay giảm về mức vô lý ( Dữ liệu không bị sai và lỗi)
# 
# 2,Biến động nằm trong tầm kiểm soát và không phải do các biến động tức thời
# 
# 

# %% [markdown]
# # VẼ BIỂU ĐỒ HIỆP PHƯƠNG SAI VÀ ĐỘ TƯƠNG QUAN GIỮA CÁC CỔ PHIẾU

# %%
# Tạo ma trận hiệp phương sai theo lợi nhuận hàng ngày
cov_matrix= portfolio_data.pct_change().dropna().cov()

# Tạo ma trận tương quan theo lợi nhuận hàng ngày
corr_matrix= portfolio_data.pct_change().dropna().corr()

# Vẽ biểu đồ
fig,ax = plt.subplots(1,2,figsize=(12,4))

sns.heatmap(cov_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.6f',
            ax=ax[0]
            )
ax[0].set_title('Covariance Matrix')

sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            ax=ax[1]
            )
ax[1].set_title('Correlation Matrix')

plt.tight_layout()
plt.show()

# %% [markdown]
# Nhận xét:
# 
# 1, Các hệ số tương quan nằm trong khoảng từ 0.36 đến 0.56 => Đây là mức tương quan trung bình và rất phù hợp để đa dạng hoá danh mục đầu tư , các cổ phiếu không quá phụ thuộc nhau vào ngắn hạn
# 
# 2, ACGL và LOW có hệ số tương quan là 0.36 dựa trên lợi nhuận hàng ngày cho thấy mô hình Makorwitz có thể phân bổ đáng kể vào cặp đôi này để tối ưu Sharpe Ratio

# %% [markdown]
# 

# %% [markdown]
# ## FEATURE ENGINEERING

# %%
# Tính daily returns
daily_returns = portfolio_data.pct_change().dropna()
daily_returns.head(5)

# %% [markdown]
# Năm hoá:
# 
# Lợi nhuận theo năm = Lợi nhuận theo ngày * 252
# 
# Độ lệch chuẩn theo năm = Độ lệch chuẩn theo ngày * sqrt(252)
# 
# ( Giả sử có 252 phiên giao dịch trong năm )
# 

# %%
daily_mean = daily_returns.mean()
daily_std = daily_returns.std()

# Năm hoá daily returns và std của nó
annual_mean = daily_mean * 252
annual_volatility = daily_std * np.sqrt(252)

# Tính Sharpe Ratio = Lợi nhuận theo năm / Rủi ro theo năm
sharpe_ratio = annual_mean / annual_volatility

# Bảng so sánh
comparison_table = pd.DataFrame({
    'Annual Mean': annual_mean,
    'Annual Volatility': annual_volatility,
    'Sharpe Ratio': sharpe_ratio
})

comparison_table.sort_values(by='Sharpe Ratio',ascending=False)
comparison_table




# %% [markdown]
# Nhận xét:
# 
# 1, ACGL có Sharpe Ratio cao nhất và là cổ có hiệu quả tốt nhất danh mục
# 
# 2, FITB tuy có lợi nhuận trung bình hằng năm cao nhất nhưng do biến động quá cao nên làm Sharpe Ratio giảm đáng kể
# 
# 3, IEX có lợi nhuận mức trung bình 12,78% nhưng lại có độ biến động thấp nhất trong 5 cổ phiếu và sẽ là sự lựa chọn an toàn nhất trong rổ khi thị trường biến động

# %% [markdown]
# ## MODELLING

# %% [markdown]
# # MÔ PHỎNG 20000 KỊCH BẢN CHIA DANH MỤC KHÁC NHAU (MONTE CARLO)

# %%
# Định nghĩa các tham số đầu vào
exp_returns = annual_mean
cov_matrix = daily_returns.cov()*252   # Ma trận hiệp phương sai
num_assets = len(exp_returns)     # Số lượng mã cổ phiếu
num_simulations = 20000           # Số lượng kịch bản xây dựng

# Ma trận lưu kết quả
results=np.zeros((num_assets+3,num_simulations))

for i in range(num_simulations):
    # Tạo trọng số ngẫu nhiên tương ứng với tỷ lệ nắm giữ của 5 cổ phiếu
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    # Tính lợi nhuận của danh mục
    portfolio_return = np.sum(exp_returns * weights)

    # Tính rủi ro danh mục
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    #Lưu kết quả
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    results[2,i] = results[0,i] / results[1,i] # Sharpe Ratio
    for j in range(num_assets):
        results[j+3,i]=weights[j]

columns=['Returns','Volatility','Sharpe Ratio']+[ticker for ticker in exp_returns.index]
results_df = pd.DataFrame(results.T,columns=columns)
results_df.head(5)



# %%
# Tìm 2 kịch bản chia portfolio đem lại Sharpe Ratio cao nhất và Volatility thấp nhất
max_sharpe_ratio = results_df.iloc[results_df['Sharpe Ratio'].idxmax()]
min_vol_ratio = results_df.iloc[results_df['Volatility'].idxmin()]

print(max_sharpe_ratio,min_vol_ratio,sep='\n\n')

# %% [markdown]
# # VẼ ĐƯỜNG BIÊN HIỆU QUẢ

# %%
plt.figure(figsize=(12, 8))
# Vẽ tất cả các danh mục mô phỏng
plt.scatter(results_df.Volatility, results_df.Returns, c=results_df['Sharpe Ratio'], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')

# Đánh dấu danh mục có Max Sharpe (Sao đỏ)
plt.scatter(max_sharpe_ratio.Volatility, max_sharpe_ratio.Returns, color='r', marker='*', s=200, label='Max Sharpe Ratio')

# Đánh dấu danh mục có Min Volatility (Sao xanh)
plt.scatter(min_vol_ratio.Volatility, min_vol_ratio.Returns, color='b', marker='*', s=200, label='Min Volatility')

plt.title('Danh mục hiệu quả - Giả lập Monte Carlo')
plt.xlabel('Biến động theo năm')
plt.ylabel('Lợi nhuận trung bình hằng năm')
plt.legend()
plt.show()

# %% [markdown]
# Nhận xét:
# 
# 1, Ngôi sao đỏ chính là vùng đầu tư thông minh nhất dựa trên chỉ số Sharpe Ratio đem lại mức bù đắp lợi nhuận lớn nhất dựa trên mỗi đơn vị rủi ro (18%)
# 
# 2, Ngôi sao xanh chính là vùng an toàn nhất với độ biến động thấp nhất nhưng bù lại lợi nhuận đem lại chỉ ở mức 14%
# 
# 3, Đường cánh cung được nối từ sao đỏ với sao xanh chính là vùng đầu tư đối với rổ cố phiếu này khi với mỗi điểm ở cánh cung này thì không thể có điểm có lợi nhuận cao hơn mà rủi ro không tăng

# %% [markdown]
# ## SO SÁNH GIỮA VIỆC SỬ DỤNG BI-LSTM VÀ XGBOOST ĐỂ DỰ ĐOÁN XU HƯỚNG TĂNG/GIẢM VÀO NGÀY TIẾP THEO ( ACGL )

# %% [markdown]
# # 1, FEATURE ENGINEERING

# %%
data=data[data['ticker']=='ACGL']
data.head(5)

# %%
!pip install pandas_ta

# %%
import pandas_ta as ta

# Return
daily_return=data.Close.pct_change()
data['Log_Daily_Return'] =np.log(1+daily_return)
data['Return']=daily_return

lag_log_return=data['Log_Daily_Return'].shift(1)
data['Lag_Log_Return']=lag_log_return

# Volume
data['Volume_Change'] = data.groupby('ticker').Volume.pct_change()
volume_ma_20 =data.groupby('ticker').Volume.transform(lambda x : x.rolling(20).mean())
data['Volume_Change_MA20'] = data.Volume / volume_ma_20

# RSI
data['RSI_14'] = data.groupby('ticker').Close.transform(lambda x : ta.rsi(x,length=14))
data['RSI_14_MA60']=data.groupby('ticker')['RSI_14'].transform(lambda x : x.rolling(60).mean())

# EMA
data['EMA_20'] = data.groupby('ticker').Close.transform(lambda x :x.ewm(span=20,adjust=False).mean())
data['EMA_Cross'] = data['EMA_20']-data.groupby('ticker').Close.transform(lambda x :x.ewm(span=60,adjust=False).mean())
data['Dist_EMA_60'] = (data['Close'] - data.groupby('ticker').Close.transform(lambda x :x.ewm(span=60,adjust=False).mean())) / data.groupby('ticker').Close.transform(lambda x :x.ewm(span=60,adjust=False).mean())

# Volatility
data['Rolling_Std_20'] = data.groupby('ticker').Close.transform(lambda x : x.rolling(20).std())
data['H_L_Spread']= (data['High'] - data['Low']) / data['Close']

# Tạo nhãn : 1 nếu ngày mai tăng còn 0 nếu ngày mai giảm
data['Target_Class']=np.where(data['Log_Daily_Return'].shift(-1)>0 ,1,0 )


data.dropna(inplace=True)
data.head(5)

# %% [markdown]
# # 2, XÂY DỰNG VÀ ĐÁNH GIÁ MÔ HÌNH HỌC SÂU (BI-LSTM)

# %%
from sklearn.preprocessing import MinMaxScaler,LabelEncoder



features=['Volume_Change','Volume_Change_MA20','RSI_14','RSI_14_MA60','EMA_20','EMA_Cross','Rolling_Std_20','H_L_Spread']


# Chia dữ liệu thành train và test
final_data=data.dropna().copy()
train_size= int(len(final_data)*0.8)
train_data=final_data.iloc[:train_size]
test_data=final_data.iloc[train_size:]

# Scale dữ liệu về khoảng 0,1
scaler_x = MinMaxScaler(feature_range=(0,1))

train_data[features] = scaler_x.fit_transform(train_data[features])
test_data[features] = scaler_x.transform(test_data[features])

# Tạo cấu trúc chuỗi cho LSTM
def create_sequences(df,length):
    x = []
    y = []
    for ticker in df.ticker.unique():
        # Lấy dữ liệu riêng cho từng mã cố phiếu
        ticker_df = df[df['ticker'] == ticker]
        ticker_values = ticker_df[features + ['Target_Class']].values
        for i in range(length ,len(ticker_values)):
            x.append(ticker_values[i-length:i, : -1])
            y.append(ticker_values[i,-1])
    return np.array(x),np.array(y)

x_train ,  y_train = create_sequences(train_data,60)
x_test  ,  y_test  = create_sequences(test_data,60)


# %%
x_train.shape

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,BatchNormalization,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

model = Sequential()

# Layer 1
model.add(Bidirectional(LSTM(units=20, return_sequences=True,kernel_regularizer=regularizers.l2(0.01), input_shape=(x_train.shape[1], x_train.shape[2]))))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# Layer2
model.add(Bidirectional(LSTM(units=32)))
model.add(Dropout(0.35))

model.add(Dense(units=1,activation='sigmoid'))


# %%
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# %%
from sklearn.utils import class_weight

# Tính toán trọng số dựa trên phân phối thực tế của y_train
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(weights))

early_stopping = EarlyStopping (monitor= 'val_loss',
                                patience=10,
                                restore_best_weights=True
                                )
model.fit(x_train,y_train,epochs = 50,batch_size = 32, callbacks=early_stopping,validation_split=0.1,verbose=1,shuffle=False,class_weight=class_weights_dict)

# %% [markdown]
# # VẼ BIỂU ĐỒ SO SÁNH THỰC TẾ VÀ DỰ ĐOÁN

# %%
import matplotlib.pyplot as plt

# 1. Lấy xác suất dự báo từ mô hình
y_pred_probs = model.predict(x_test)

# 2. Chuyển xác suất thành nhãn 0/1
y_pred_labels = (y_pred_probs > np.median(y_pred_probs)).astype(int).flatten()

# 3. Vẽ biểu đồ so sánh 50 điểm dữ liệu đầu tiên
plt.figure(figsize=(10, 6))
plt.plot(y_test[:50], label='Thực tế (1: Tăng, 0: Giảm)', marker='o', linestyle='', alpha=0.6)
plt.plot(y_pred_labels[:50], label='Dự báo', marker='x', linestyle='', color='red')
plt.xlabel('Thời gian')
plt.ylabel('Nhãn')
plt.title('So sánh thực tế và dự báo')
plt.legend()
plt.show()

# %% [markdown]
# # KIỂM TRA PHÂN PHỐI GIỮA CÁC NHÃN

# %%
# Kiểm tra giá trị xác suất thực tế mà model trả về
sns.histplot(y_pred_probs, kde=True)
plt.title("Phân phối xác suất ở trong phần dự báo")
plt.show()

# Đếm số lượng nhãn mỗi loại
print(pd.Series(y_pred_labels).value_counts())

# %% [markdown]
# # VẼ BIỂU ĐỒ SO SÁNH LỢI NHUẬN GIỮA VIỆC SỬ DỤNG BI-LSTM VÀ CHIẾN THUẬT MUA RỒI GIỮ

# %%

# Tạo bảng
y_test_returns = test_data['Log_Daily_Return'].values[60:]
results = pd.DataFrame({
    'Actual_Return': y_test_returns.flatten(),
    'Prob_Up': y_pred_probs.flatten()
})

# Sử dụng ngưỡng
results['Signal'] = (results['Prob_Up'] > np.median(results['Prob_Up'])).astype(int)

# Tín hiệu ở ngày hôm nay là cơ sở cho ngày mai quyết định mua hay đứng ngoài (1 sẽ mua và tính lợi nhuận, 0 sẽ đứng ngoài quan sát)
results['Strategy_Return'] = results['Signal'].shift(1) * results['Actual_Return']
results.dropna(inplace=True)

# Tính lợi nhuận cộng dồn (quy đổi từ Log Return sang tỷ lệ phần trăm)
results['Market_Cum'] = results['Actual_Return'].cumsum().apply(np.exp)
results['Strategy_Cum'] = results['Strategy_Return'].cumsum().apply(np.exp)

# Tính sharpe ratio
daily_rf = 0 # Lãi suất phi rủi ro coi như = 0
sharpe = np.sqrt(252) * (results['Strategy_Return'].mean() - daily_rf) / results['Strategy_Return'].std()

# Max Drawdown (Mức sụt giảm tối đa từ đỉnh)
peak = results['Strategy_Cum'].cummax()
drawdown = (results['Strategy_Cum'] - peak) / peak
max_drawdown = drawdown.min()

# Vẽ biểu đồ
plt.figure(figsize=(17, 7))
plt.plot(results['Market_Cum'], label='Thị trường (Mua rồi giữ)', color='gray', alpha=0.5, linestyle='--')
plt.plot(results['Strategy_Cum'], label='Bi-LSTM Strategy', color='blue', linewidth=2)

plt.title(f'KẾT QUẢ BACKTEST\nSharpe Ratio: {sharpe:.2f} | Max Drawdown: {max_drawdown*100:.2f}%', fontsize=14)
plt.xlabel('Thời gian (Số phiên)')
plt.ylabel('Tỷ lệ sinh lời dựa trên tài sản gốc')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Mức sinh lời của thị trường: {(results['Market_Cum'].iloc[-1]-1)*100:.2f}%")
print(f"Mức sinh lời của chiến thuật: {(results['Strategy_Cum'].iloc[-1]-1)*100:.2f}%")

# %% [markdown]
# ## 3,XÂY DỰNG VÀ ĐÁNH GIÁ MÔ HÌNH HỌC MÁY (XGBOOSTCLASSIFIER)

# %%
!pip install lazypredict

# %% [markdown]
# # THỬ NGHIỆM TRÊN NHIỀU MÔ HÌNH KHÁC NHAU

# %%
from lazypredict.Supervised import LazyClassifier

features.append('Lag_Log_Return')


# Sử dụng dữ liệu đã chia
X_train_ml = train_data[features]
y_train_ml = train_data['Target_Class']
X_test_ml = test_data[features]
y_test_ml = test_data['Target_Class']

# Thử trên nhiều mô hình classification
clf= LazyClassifier(verbose = 0,ignore_warnings=True,custom_metric=None)
models , predictions = clf.fit(X_train_ml,X_test_ml,y_train_ml,y_test_ml)

#Đưa ra kết quả
models

# %%
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Khởi tạo mô hình với các tham số chống Overfitting
xgb_model = XGBClassifier(
    n_estimators=1000,         # Số lượng cây
    learning_rate=0.05,       # Tốc độ học
    max_depth=6,              # Độ sâu tối đa mỗi cây
    subsample=0.8,            # Mỗi cây sẽ chỉ được thấy 80% dữ liệu ngẫu nhiên
    colsample_bytree=0.8,     # Mỗi cây sẽ chỉ được thấy 80% features
    eval_metric='logloss',    # Đo lường sai số
    random_state=42           # Để các lần chạy kết quả giống nhau
)


# Huấn luyện
xgb_model.fit(X_train_ml, y_train_ml)

# Dự báo xác suất
y_pred_probs_ml = xgb_model.predict_proba(X_test_ml)[:, 1]

# Dùng ngưỡng Median
threshold_ml = np.median(y_pred_probs_ml)
y_pred_labels_ml = (y_pred_probs_ml > threshold_ml).astype(int)

print(classification_report(y_test_ml, y_pred_labels_ml))

# %% [markdown]
# # VẼ BIỂU ĐỒ PHÂN PHỐI NHÃN

# %%
sns.histplot(y_pred_probs_ml, kde=True)
plt.title("Phân phối xác suất ở trong phần dự báo")
plt.show()

# Đếm số lượng từng nhãn
print(pd.Series(y_pred_labels_ml).value_counts())

# %% [markdown]
# # VẼ BIỂU ĐỒ SO SÁNH MỨC ĐỘ QUAN TRỌNG CÁC FEATURES

# %%
feature_importance = pd.Series(xgb_model.feature_importances_,index=features)

# Sắp xếp các features theo độ quan trọng
feature_importance.nlargest(10).plot(kind='barh',figsize=(10,6),color='purple')
plt.title('Thứ hạng features quan trọng')
plt.xlabel('% quan trọng')
plt.show()

# %% [markdown]
# CÁC FEATURES CÓ ĐÓNG GÓP GẦN NHƯ TƯƠNG ĐƯƠNG NHAU

# %% [markdown]
# # VẼ BIỂU ĐỒ SO SÁNH LỢI NHUẬN GIỮA VIỆC SỬ DỤNG XGBOOST VÀ CHIẾN THUẬT MUA RỒI GIỮ

# %%
y_test_ml_returns = test_data['Log_Daily_Return'].values[len(test_data) - len(y_pred_probs_ml) :]
res = pd.DataFrame({
    'Actual_Return': y_test_ml_returns.flatten(),
    'Prob_Up': y_pred_probs_ml.flatten()
})

# Sử dụng ngưỡng là median để tạo tín hiệu mua hay đứng ngoài
res['Signal'] = (res['Prob_Up'] > np.median(res['Prob_Up'])).astype(int)

# Tín hiệu ngày hôm qua là cơ sở cho quyết định ngày hôm nay
res['Strategy_Return'] = res['Signal'].shift(1) * res['Actual_Return']
res.dropna(inplace=True)

# Tính lợi nhuận cộng dồn (quy đổi từ Log Return sang tỷ lệ phần trăm)
res['Market_Cum'] = res['Actual_Return'].cumsum().apply(np.exp)
res['Strategy_Cum'] = res['Strategy_Return'].cumsum().apply(np.exp)

# Tính sharpe ratio
daily_rf = 0 # Lãi suất phi rủi ro coi như = 0
sharpe = np.sqrt(252) * (res['Strategy_Return'].mean() - daily_rf) / res['Strategy_Return'].std()

# Max Drawdown (Mức sụt giảm tối đa từ đỉnh)
peak = res['Strategy_Cum'].cummax()
drawdown = (res['Strategy_Cum'] - peak) / peak
max_drawdown = drawdown.min()

# Vẽ biểu đồ
plt.figure(figsize=(17, 7))
plt.plot(res['Market_Cum'], label='Thị trường (Mua rồi giữ)', color='gray', alpha=0.5, linestyle='--')
plt.plot(res['Strategy_Cum'], label='XGBOOST', color='blue', linewidth=2)

plt.title(f'KẾT QUẢ BACKTEST\nSharpe Ratio: {sharpe:.2f} | Max Drawdown: {max_drawdown*100:.2f}%', fontsize=14)
plt.xlabel('Thời gian (Số phiên)')
plt.ylabel('Tỷ lệ sinh lời dựa trên tài sản gốc')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Mức sinh lời của thị trường: {(res['Market_Cum'].iloc[-1]-1)*100:.2f}%")
print(f"Mức sinh lời của chiến thuật: {(res['Strategy_Cum'].iloc[-1]-1)*100:.2f}%")



# %% [markdown]
# ## 4,KẾT LUẬN

# %% [markdown]
# Mô hình Bi-LSTM: Nhờ vào kiến trúc mạng nơ-ron hồi quy hai chiều, mô hình có khả năng khai thác tối ưu tính phụ thuộc thời gian (time-dependency) và các mẫu hình phi tuyến tính phức tạp trong dữ liệu lịch sử. Khả năng "ghi nhớ" dài hạn giúp Bi-LSTM bắt kịp các điểm đảo chiều xu hướng một cách khá nhạy bén với minh chứng là accuaracy của tập val lên đến hơn 57%.Tuy nhiên ,do quy mô dữ liệu quá nhỏ nên mô hình rất dễ bị overfitting và đây cũng không phải là cách tiếp cận tối ưu nhất.

# %% [markdown]
# Mô hình XGBoost: Là một thuật toán mạnh mẽ dựa trên Boosting cây quyết định, nhưng đối với dữ liệu chuỗi thời gian tài chính có độ nhiễu cao, XGBoost dễ rơi vào tình trạng quá khớp (overfitting) hoặc phản ứng chậm với các biến động ngắn hạn nếu thiếu các đặc trưng kỹ thuật chuyên sâu.Tuy nhiên, mô hình này vẫn có ứng dụng trong thực tiễn cao


