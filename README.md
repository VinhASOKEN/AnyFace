# Anyface - AI Hackathon Face Analyst Challenge Project 
Project được thực hiện và duy trì bởi [Nguyễn Thành Vinh](https://github.com/VinhASOKEN), [Nguyễn Hồng Vân]() và [Phan Vũ Nguyên Hoàng](https://github.com/t1tc01).

# Giới thiệu 

# Project
 Code train và weights đã được public trong folder của từng model. <br>
  ## Training 
  Setup dữ liệu training - đọc trong folder data. <br>
   
  ## Inference
  Chạy file infer.py để thử nghiệm kết quả các model.<br>
  Lưu ý thay đổi đường dẫn phù hợp trong các file config và train.<br>

  ## Demo 
![](https://github.com/VinhASOKEN/AnyFace/blob/main/result_images/967456.jpg)
```
BBox            : 391.5874938964844, 429.3175048828125, 563.9922790527344, 684.9896240234375
Class Age       : 20-30s
Class Emotional : Happiness
Class Gender    : Female
Class Mask      : unmasked
Class Race      : Mongoloid
Class Skintone  : light
```
![](https://github.com/VinhASOKEN/AnyFace/blob/main/result_images/968261.jpg)
```
BBox            : 322.0959167480469, 298.181640625, 595.6968078613281, 855.4468994140625
Class Age       : 20-30s
Class Emotional : Neutral
Class Gender    : Male
Class Mask      : masked
Class Race      : Mongoloid
Class Skintone  : light
```

Ngoài ra model detect face còn trả về 5 điểm landmark khuôn mặt, xem chi tiết tại: detect_face/tools/detector.py<br>
  

  

