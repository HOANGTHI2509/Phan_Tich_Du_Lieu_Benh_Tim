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
