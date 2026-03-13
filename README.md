# S&P 500 Financial Analyst: Stock Trend Prediction 📈

Dự án sử dụng các mô hình học máy và học sâu (Machine Learning & Deep Learning) để phân tích dữ liệu lịch sử của chỉ số S&P 500 và dự báo xu hướng thị trường.

## 📌 Tổng quan dự án (Project Overview)
Dự án tập trung vào việc thu thập dữ liệu tài chính 10 năm của S&P 500 từ Kaggle, sau đó thực hiện kỹ thuật tiền xử lý dữ liệu và xây dựng các mô hình dự báo để so sánh hiệu quả giữa các thuật toán khác nhau.

### Các tính năng chính:
* **Data Acquisition:** Tải dữ liệu tự động thông qua thư viện `kagglehub`.
* **Backtesting:** Hệ thống tự động kiểm thử chiến lược trên dữ liệu lịch sử.
* **Model Comparison:** So sánh trực quan giữa mô hình truyền thống (XGBoost) và mô hình học sâu (Bi-LSTM).
* **Performance Metrics:** Đánh giá dựa trên lợi nhuận thực tế (Cumulative Return), chỉ số Sharpe và Max Drawdown.

## 🛠 Công nghệ sử dụng (Tech Stack)
* **Ngôn ngữ:** Python 3.x
* **Thư viện phân tích dữ liệu:** Pandas, Numpy
* **Trực quan hóa:** Matplotlib, Plotly
* **Học máy & Học sâu:** * `TensorFlow/Keras` (cho mô hình Bi-LSTM)
    * `XGBoost`
    * `Scikit-learn`
* **Môi trường:** Google Colab / VS Code

## 📊 Kết quả thực nghiệm (Experimental Results)
Dựa trên quá trình thử nghiệm, dự án đã rút ra các kết luận quan trọng:
* **Bi-LSTM:** Đạt tỷ lệ sinh lời cao nhất (**16.17%**) với sự ổn định vượt trội so với thị trường (Buy & Hold).
* **XGBoost:** Hiệu quả thấp hơn trong việc bắt các chuỗi thời gian dài nhưng có tốc độ huấn luyện nhanh.
