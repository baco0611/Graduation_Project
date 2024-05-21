# KHẢO SÁT TẬP DỮ LIỆU NGÔN NGỮ KÝ HIỆU SỐ

## Giới thiệu về dataset

Đây là bộ dữ liệu do người dùng Muhammad Khalid chia sẻ trên nền tảng Kaggle các hình ảnh về ngôn ngữ ký hiệu - ngôn ngữ hình thể dành cho người khiếm thính, cụ thể là các ký hiệu tay đại diện cho các con số từ 0 đến 9 (hình ![dataset_ex_num](figs:dataset_ex_num)). Tập dữ liệu được thiết kế với mục tiêu chính là phát triển các mô hình nhận diện ngôn ngữ ký hiệu số, một ứng dụng quan trọng trong việc hỗ trợ giao tiếp cho người khiếm thính và câm. Điều này không chỉ giúp tăng cường khả năng giao tiếp cho người bị khiếm khuyết các chức năng của cơ thể mà còn mở ra nhiều ứng dụng tiềm năng trong các lĩnh vực khác như giáo dục, dịch vụ khách hàng và hệ thống tự động hóa.

Đặc điểm của bộ dữ liệu:

- **Số lượng hình ảnh**: Tập dữ liệu bao gồm 15.000 hình ảnh, đại diện cho các ký hiệu tay của 10 con số từ 0 đến 9. 

- **Độ phân giải**: Các hình ảnh trong tập dữ liệu có độ phân giải ổn định trong khoảng từ 75px * 100px đến 100px * 100px, đảm bảo rằng các chi tiết nhỏ trên tay có thể được nhận diện rõ ràng.

- **Đa dạng về người thực hiện ký hiệu**: Các hình ảnh được chụp từ nhiều người khác nhau, tạo ra sự đa dạng trong cách thực hiện các ký hiệu tay.

- **Điều kiện ánh sáng và môi trường**: Hình ảnh được chụp trong các điều kiện ánh sáng và môi trường khác nhau, giúp mô hình có khả năng nhận diện ký hiệu trong các tình huống thực tế đa dạng.

- **Ảnh đen trắng**: Các hình ảnh trong tập dữ liệu là ảnh đen trắng, điều này không ảnh hưởng đến tính thực tiễn của phương pháp trích xuất đặc trưng SIFT, vì SIFT không phụ thuộc vào màu sắc mà tập trung vào các đặc trưng không gian của hình ảnh. Tuy nhiên, đối với CNN, việc chỉ sử dụng ảnh đen trắng có thể gây ảnh hưởng, do CNN sử dụng chính giá trị pixel của ảnh để làm đầu vào. Mặc dù vậy, điều này không ảnh hưởng đến việc đánh giá và so sánh hai mô hình, vì mục tiêu là kiểm tra khả năng phân loại của chúng trên cùng một tập dữ liệu.

## Khảo sát mô hình

### Cấu trúc ban đầu của dự án

Cấu trúc của dự án được thể hiện như bên dưới. Đây là cấu trúc ban đầu khi đã clone/download source code từ Python về máy tính, trong đó có các folder được cấu trúc như bên dưới, nhưng thiếu đi các folder chứa dữ liệu cần thiết. Vui lòng đọc tutorial này để tạo folder và run từng files theo đúng thứ tự, tránh xảy ra sai sót.

``` bash
Sign_digit
|_ dataset
|   |_ images //Nơi lưu dataset gốc, ta sẽ không chỉnh sửa hay thao tác với dữ liệu ở folder này
|_ load_data //Nơi chứa các file
|_ SIFT 
|_ VGG8
```

### Load dữ liệu

    Đầu tiên, ta cần chuẩn bị dữ liệu để có thể thực hiện quá trình khảo sát. Dataset gốc được lưu trữ tại folder ```dataset/image```, cần phải load data để tiền xử lý, tạo biến thể ảnh và nén file lại để tái sử dụng mà không cần phải duyệt toàn bộ ảnh nữa. 
    
    Tạo folder data bên trong folder dataset với path ```dataset/data``` để chứa các dữ liệu được nén lại. Đây sẽ là các dữ liệu có thể tái sử dụng mà không cần phải duyệt lại toàn bộ dataset.

    Để thực hiện việc load dữ liệu, trỏ vào folder ```dataset/image``` và chạy file "load_data.py" để load dữ liệu và nén dữ liệu.

    Sau khi chạy, các folder chứa biến thể sẽ được tạo ra bên trong folder dataset. Ngoài ta, bên trong folder data cũng có các file joblib chứa dữ liệu biến thể. Trong đó, file process chứa toàn bộ dữ liệu tổng hợp, dùng cho quá trình training mà không cần phải nối các biến thể lại. Các file biến thể có thể dùng để huấn luyện độc lập, hoặc dùng trong việc kiếm thử.

    Khởi chạy file ```testing_data.py``` nếu muốn kiếm tra xem quá trình chạy có lỗi gì không. Quá trình không lỗi là khi dữ liệu được nối vào file process đúng và tất cả giá trị đều trả về bằng 0.

### Khảo sát phương pháp trích xuất đặc trưng SIFT + BoVW

    Đầu tiên, tạo folder ```SIFT/data``` để chứa tất cả các file là dữ liệu và mô hình được nén lại cho việc tái sử dụng dữ liệu. Trong đó, tạo 2 folder con là ```SIFT/data/model``` và ```SIFT/data/dataset``` theo cấu trúc bên dưới:

    ```bash
    SIFT
    |_ data
    |   |_ dataset //chứa dữ liệu đặc trưng
    |   |_ model // chứa các model
    
    ```

    Đầu tiên, ta thực hiện bước trích xuất đặc trưng SIFT từ các dữ liệu hình ảnh đã được chuẩn bị ở bước load dữ liệu. Ta chạy chương trình tại file ```extracting_feature.py``` để trích xuất đặc trưng SIFT.