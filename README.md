# Hessian_sift
Hessian-Affine detector, SIFT descriptor  
original from : [hesaff](https://github.com/perdoch/hesaff)


## 编译特征提取接口
Makefile(linux, opencv3.2)
```
all: *.cpp
	g++ -O3 -Wall -o hesaff pyramid.cpp affine.cpp siftdesc.cpp helpers.cpp hesaff.cpp `pkg-config opencv4 --cflags --libs` -lrt

```
Makefile(mac, opencv4.1)
```
all: *.cpp
	g++ -std=c++11 -O3 -Wall -o hesaff pyramid.cpp affine.cpp siftdesc.cpp helpers.cpp hesaff.cpp `pkg-config opencv4 --cflags --libs`
```

## python测试
简单实验了原始sift和hesaff（图像宽最大限制500）
```
python demo_python.py
```
其中二进制文件(hesaff.sift)读取：[by insikk](https://github.com/perdoch/hesaff/blob/fcd04c9b7d2b7361ec676e1ded228768823eb0b3/util/Batch%20SIFT%20Extractor.ipynb)



## Citation
```
Perdoch, M. and Chum, O. and Matas, J.: Efficient Representation of
Local Geometry for Large Scale Object Retrieval. In proceedings of
CVPR09. June 2009.

TBD: A reference to technical report describing the details and some
retrieval results will be placed here.
```