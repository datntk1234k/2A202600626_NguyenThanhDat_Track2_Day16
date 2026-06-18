# Báo cáo Lab 16 - Phương án CPU thay thế GPU

## Lý do dùng CPU thay GPU
Tài khoản AWS mới bị giới hạn quota GPU = 0 vCPU cho dòng G/VT instances. Yêu cầu tăng quota không được duyệt trong thời gian làm lab. Ngoài ra, tổng vCPU limit của tài khoản là 8 vCPU, nên phải dùng r5.xlarge (4 vCPU) thay vì r5.2xlarge (8 vCPU).

## Kết quả Benchmark trên r5.xlarge (4 vCPU, 32 GB RAM)

| Metric | Kết quả |
|---|---|
| Dataset | Credit Card Fraud Detection (284,807 rows) |
| Load time | 2.40s |
| Training time | 1.49s |
| Best iteration | 3 |
| AUC-ROC | 0.9179 |
| Accuracy | 99.30% |
| F1-Score | 0.2963 |
| Precision | 0.1791 |
| Recall | 0.8571 |
| Inference latency (1 row) | 0.396 ms |
| Inference throughput (1000 rows) | 0.695 ms |

## Nhận xét
- LightGBM train rất nhanh (1.49s) trên CPU với dataset 284K rows nhờ thuật toán gradient boosting hiệu quả.
- AUC-ROC 0.917 cho thấy model phân biệt tốt giao dịch gian lận.
- Recall cao (0.857) phù hợp bài toán fraud detection — bắt được phần lớn gian lận.
- Instance r5.xlarge (~$0.252/giờ) không cần quota đặc biệt, phù hợp tài khoản mới.
