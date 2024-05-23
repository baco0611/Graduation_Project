# TÌM HIỂU CÁC PHƯƠNG PHÁP TRÍCH XUẤT ĐẶC TRƯNG TRONG BÀI TOÁN PHÂN LOẠI HÌNH ẢNH

## Giới thiệu về dự án

Đây là dự án khóa luận tốt nghiệp được thực hiện bởi sinh viên Huỳnh Văn Nguyên Bảo, ngành Công nghệ thông tin, chuyên ngành Khoa học máy tính, trường Đại học Khoa học, Đại học Huế. Mục tiêu của dự án là nghiên cứu và so sánh các phương pháp trích xuất đặc trưng trong bài toán phân loại hình ảnh, đặc biệt tập trung vào hai phương pháp chính: SIFT (Scale-Invariant Feature Transform) và mô hình mạng nơ-ron tích chập (Convolutional Neural Network - CNN) VGG8.

Mục tiêu chính của dự án không chỉ dừng lại ở việc nắm vững kiến thức lý thuyết về các mô hình mà còn là áp dụng nó vào thực tế để tạo ra giá trị thực sự. Hy vọng rằng thông qua dự án này có thể cung cấp các kiến thức bổ ích về hai phương pháp trích xuất đặc trưng trên trong cho các nghiên cứu sau này.


## Giới thiệu về dataset

### Dataset ngôn ngữ kí hiệu số
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

- Đầu tiên, ta cần chuẩn bị dữ liệu để có thể thực hiện quá trình khảo sát. Dataset gốc được lưu trữ tại folder ```dataset/image```, cần phải load data để tiền xử lý, tạo biến thể ảnh và nén file lại để tái sử dụng mà không cần phải duyệt toàn bộ ảnh nữa. 
    
- Tạo folder data bên trong folder dataset với path ```dataset/data``` để chứa các dữ liệu được nén lại. Đây sẽ là các dữ liệu có thể tái sử dụng mà không cần phải duyệt lại toàn bộ dataset.

- Để thực hiện việc load dữ liệu, trỏ vào folder ```dataset/image``` và chạy file "load_data.py" để load dữ liệu và nén dữ liệu.

- Sau khi chạy, các folder chứa biến thể sẽ được tạo ra bên trong folder dataset. Ngoài ta, bên trong folder data cũng có các file joblib chứa dữ liệu biến thể. Trong đó, file process chứa toàn bộ dữ liệu tổng hợp, dùng cho quá trình training mà không cần phải nối các biến thể lại. Các file biến thể có thể dùng để huấn luyện độc lập, hoặc dùng trong việc kiếm thử.

- Khởi chạy file ```testing_data.py``` nếu muốn kiếm tra xem quá trình chạy có lỗi gì không. Quá trình không lỗi là khi dữ liệu được nối vào file process đúng và tất cả giá trị đều trả về bằng 0.

### Khảo sát phương pháp trích xuất đặc trưng SIFT + BoVW (SIFT)

- Đầu tiên, tạo folder ```SIFT/data``` để chứa tất cả các file là dữ liệu và mô hình được nén lại cho việc tái sử dụng dữ liệu, folder ```SIFT/image``` để chứa các file ảnh là confusion matrix được tạo ra trong quá trình khảo sát. Trong đó, tạo 2 folder con là ```SIFT/data/model``` và ```SIFT/data/dataset``` theo cấu trúc bên dưới:

    ```bash
    SIFT
    |_ data
    |   |_ dataset //chứa dữ liệu đặc trưng
    |   |_ model // chứa các model
    |_ function
    |_ image
    |_ ...
    
    ```

- Ta thực hiện bước trích xuất đặc trưng SIFT từ các dữ liệu hình ảnh đã được chuẩn bị ở bước load dữ liệu. Ta chạy chương trình tại file ```extracting_feature.py``` để trích xuất đặc trưng SIFT.

- Tiếp đến, chạy file ```training_codebook.py``` để tạo ra từ điển codebook với mô hình phân cụm K-means - đây chính là ý tưởng về mô hình BoVW (Bags of Visual Word). 

- Các tham số được điều chỉnh trong file config. Trong đó:
    - size: kích thước từ điển muốn huấn luyện
    - name: tập dữ liệu muốn sử dụng (raw: ảnh gốc, negative: ảnh âm, resized: ảnh méo, rotated: ảnh xoay, flipped: ảnh lật, process: toàn bộ các dữ liệu trước)
    - date: để tránh việc ghi đè các mô hình với nhau khi huấn luyện, vui lòng điền ngày tháng theo định dạng "yyyymmdd". Nếu trong một ngày, huấn luyện nhiều mô hình thì thêm hậu tố số thứ tự theo định dạng "yyyymmdd_n.
    - Ví dụ với size 200, dataset process, training model thứ 2 ngày 20240520 thì điều chỉnh giá trị như bên dưới:
    ```python
        size = 200
        name = "process"
        date = "20240520_2"
    ```

- Để huấn luyện mô hình phân loại hình ảnh, vui lòng chạy file ```training_model.py```. Để lựa chọn mô hình codebook, hãy thay đổi các giá trị trong file ```config.py```. Ngoài ra, giá trị ``date_svm` cũng cần được thay đổi với ý nghĩa và cú pháp tương tự.

- Cuối cùng, để thực hiện việc kiếm thử mô hình với các tiêu chí hình ảnh bị biến đổi, chạy file ```validation.py```

### Khảo sát mô hình VGG8 (VGG8)

- Tương tự với khi khảo sát SIFT, ta cũng cần tạo các folder để lưu dữ liệu với cấu trúc:
    ```bash
    VGG8
    |_ function
    |   |_ Processing_function.py // Chứa các hàm dùng chung
    |   |_ config.py // Chứa các tham số điều chỉnh
    |_ VGG8
        |_ model
        |_ image
        |_ ...
    ```

- Điều chỉnh các thông tin trong file ```config.py``` với các thông số:
    - data_num: số thứ tự dataset muốn dùng (0: ảnh gốc, 1: ảnh âm, 2: ảnh méo, 3: ảnh xoay, 4: ảnh lật, 5: toàn bộ các dữ liệu trước)
    - date: quy ước và ý nghĩa như SIFT
    - num_of_epoch: số epoch muốn thực hiện

- Chạy file ```training_model.py``` để thực hiện quá trình huấn luyện mô hình VGG8.

- Chạy file ```training_valid.py``` để kiếm tra độ chính xác của mô hình trên toàn bộ tập dữ liệu sử dụng (training set + valid set). Vì mô hình VGG8 có số lượng neural và dữ liệu của bộ dữ liệu môn ngữ kí hiệu số khá lớn, nên tránh việc tràn bộ nhớ, tôi đã tách riêng việc kiếm thử trên toàn bộ tập dữ liệu thành một chương trình riêng.

- **Lưu ý:** Vì cấu hình của mỗi máy là khác nhau, vì thế hãy điều chỉnh cách biến batch_size cho hợp lý với cấu hình của máy tránh việc tràn bộ nhớ trong quá hình huấn luyện và kiếm thử.

- Cuối cùng, hãy chạy file ``validation.py` để tiến hành kiểm thử và đánh giá trên các tiêu chí.

### Khảo sát mô hình SVM được huấn luyện bởi các vector đặc trưng được trích xuất từ VGG8 (VGG8_SVM)

- Đầu tiên, tạo các folder để lưu dữ liệu với cấu trúc bên dưới:

    ```bash
    VGG8
    |_ function
    |_ VGG8
    |_ VGG8_SVM
        |_ data
        |   |_ dataset
        |   |_ model
        |_ image

    ```

- Ở phần này, ta sẽ dùng các vector được trích xuất ở các lớp fully connected layer của mạng VGG8 để làm vector đặc trưng của hình ảnh (cấu trúc như bên dưới). Để trích xuất đặc trưng của hình ảnh, chạy file ```exctract_feature.py```. ***Hãy nhớ chỉnh tham số ở file ```config.py``` để chọn đúng mô hình mong muốn*** 

    ```bash
        0 conv2d (None, 224, 224, 16)
        1 max_pooling2d (None, 112, 112, 16)
        2 conv2d_1 (None, 112, 112, 32)
        3 max_pooling2d_1 (None, 56, 56, 32)
        4 conv2d_2 (None, 56, 56, 64)
        5 max_pooling2d_2 (None, 28, 28, 64)
        6 conv2d_3 (None, 28, 28, 128)
        7 max_pooling2d_3 (None, 14, 14, 128)
        8 conv2d_4 (None, 14, 14, 128)
        9 max_pooling2d_4 (None, 7, 7, 128)
        10 flatten (None, 6272)
        11 dense (None, 4096)
        12 dropout (None, 4096)
        13 dense_1 (None, 1024)
        14 dropout_1 (None, 1024)
        15 dense_2 (None, n) //n là số units ứng với số lớp của bộ dữ liệu
    ```

**Lưu ý:** Mỗi lần training thì nên chỉnh sửa các thông số ở file ```function/config.py``` để đảm bảo đúng mô hình VGG8 dùng để trích xuất và lặp lại quá trình trích xuất đặc trưng.

- Sau khi đã trích xuất đặc trưng từ mạng VGG8, tiếp tục huấn luyện mô hình SVM sử dụng các vector đặc trưng đó. Trước hết, cần chỉnh lại các tham số trong file ```VGG8_SVM_config.py```, các thông số cụ thể như sau:
    - feature_dims: số chiều dữ liệu cần trích xuất (4096 hoặc 1024)
    - date: quy ước như các phần khảo sát trên
    - data_num: quy ước như các phần khảo sát trên

- Sau đó, chạy file ```training_model.py``` để tiến hành huấn luyện mô hình SVM.

- Sau khi đã huấn luyện mô hình xong thì đánh giá mô hình với các tiêu chí bằng cách chạy file ```validation.py```

### Khảo sát mô hình SVM được huấn luyện bởi các vector đặc trưng từ VGG8 được giảm chiều bằng mô hình PCA (VGG8_PCA_SVM)



