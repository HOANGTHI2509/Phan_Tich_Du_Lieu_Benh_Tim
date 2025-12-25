# Phân Tích Dữ Liệu Bệnh Tim: Luật Kết Hợp & Phân Cụm
**(Heart Disease Analysis: Association Rules & Clustering)**

## 📖 Giới thiệu
Dự án này thực hiện quy trình Khai phá dữ liệu (Data Mining) trên tập dữ liệu tim mạch (`HeartDiseaseTrain-Test.csv`). Mục tiêu là phát hiện các mẫu tiềm ẩn giữa các triệu chứng và phân nhóm bệnh nhân dựa trên các chỉ số y sinh quan trọng.

Quy trình áp dụng hai kỹ thuật chính:
1.  **Luật kết hợp (Association Rules - Apriori):** Tìm mối liên hệ giữa các đặc điểm lâm sàng và khả năng mắc bệnh.
2.  **Phân cụm (Clustering - K-Means):** Phân nhóm bệnh nhân dựa trên đặc tính số học (Tương tự mô hình RFM).

---

## 📋 Mục lục
1. [Yêu cầu cài đặt](#-yêu-cầu-cài-đặt)
2. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
3. [Quy trình xử lý chi tiết](#-quy-trình-xử-lý-chi-tiết)
4. [Kết quả & Đánh giá](#-kết-quả--đánh-giá)

---

## 🛠 Yêu cầu cài đặt

Để chạy mã nguồn, bạn cần cài đặt Python 3.x và các thư viện hỗ trợ sau:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend

📂 Cấu trúc thư mục
Đảm bảo file dữ liệu và code nằm cùng một thư mục:

Plaintext

├── HeartDiseaseTrain-Test.csv      # [Input] File dữ liệu gốc
├── main_analysis.py                # [Script] Code Python phân tích chính
├── HeartDisease_Final_Analysis.csv # [Output] Kết quả sau khi phân cụm
└── README.md                       # Tài liệu hướng dẫn này


##🚀 Quy trình xử lý chi tiết
Chương trình thực hiện tuần tự 5 bước sau:

1. Nguồn dữ liệu (Data Loading)
Sử dụng thư viện glob để tự động dò tìm file .csv trong thư mục.

Thực hiện EDA nhanh (Exploratory Data Analysis) để kiểm tra kích thước và kiểu dữ liệu.

2. Tiền xử lý (Data Cleaning)
Đây là bước quan trọng nhất để đảm bảo độ chính xác:

Xử lý trùng lặp (Remove Duplicates): Dữ liệu gốc chứa lượng lớn bản ghi bị trùng lặp (~70%). Chương trình sẽ tự động phát hiện và loại bỏ để tránh làm sai lệch kết quả thống kê.

Xử lý giá trị thiếu (Handle Missing Values): Loại bỏ các dòng chứa giá trị null.

3. Tìm Luật kết hợp (Association Rules - Apriori)
Mục đích: Trả lời câu hỏi "Những triệu chứng nào thường đi cùng nhau dẫn đến bệnh tim?"

Kỹ thuật:

Binning: Chuyển đổi các biến số liên tục (age, cholestoral...) thành các khoảng giá trị (Ví dụ: Tuổi -> Young, Middle, Senior).

One-Hot Encoding: Mã hóa dữ liệu phân loại.

Tham số:

min_support = 0.1: Chỉ xét các tổ hợp xuất hiện trên 10%.

min_threshold (Lift) = 1.2: Chỉ lấy các luật có độ nâng > 1.2 (có ý nghĩa thống kê mạnh).

4. Phân cụm (Clustering - Method A)
Phương pháp: K-Means Clustering.

Đặc trưng đầu vào (Features): Sử dụng 5 chỉ số số học quan trọng (Numeric Features) để phân nhóm:

Age (Tuổi)

Resting Blood Pressure (Huyết áp nghỉ)

Cholestoral (Mỡ máu)

Max Heart Rate (Nhịp tim tối đa)

Oldpeak (Độ chênh ST)

Chuẩn hóa: Sử dụng StandardScaler để đưa dữ liệu về cùng một miền giá trị.

Tối ưu hóa K: Tự động chạy thử nghiệm từ K=2 đến K=6 và chọn K tốt nhất dựa trên Silhouette Score.

5. Đánh giá & Trực quan hóa (Evaluation)
PCA Visualization: Giảm chiều dữ liệu xuống 2D để vẽ biểu đồ phân tán (Scatter Plot), giúp nhìn thấy sự phân tách giữa các cụm.

Silhouette Analysis: Biểu đồ điểm số để đánh giá độ tách biệt của các cụm.

##📊 Kết quả & Đánh giá
Sau khi chạy chương trình, bạn sẽ nhận được:

1. Báo cáo trên màn hình (Console Output)
Danh sách Top 5 Luật kết hợp mạnh nhất (dựa trên chỉ số Lift).

Số cụm tối ưu (Best K).

Bảng thống kê giá trị trung bình (Mean Profile) của từng cụm, giúp định danh nhóm khách hàng (Ví dụ: Nhóm nguy cơ cao vs Nhóm khỏe mạnh).

2. Biểu đồ trực quan
Biểu đồ cột thể hiện Silhouette Score qua các giá trị K.

Biểu đồ phân tán (Scatter Plot) thể hiện các cụm trong không gian PCA.

3. File kết quả (.csv)
File HeartDisease_Final_Analysis.csv được xuất ra, chứa dữ liệu sạch và cột Cluster (nhãn cụm) để phục vụ các phân tích tiếp theo.

Dự án phục vụ mục đích học tập và nghiên cứu Khoa học dữ liệu.
