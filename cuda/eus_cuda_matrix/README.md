## Document
copy matrixMul.cu in
https://github.com/NVIDIA/cuda-samples/tree/master/Samples/matrixMul

## Build
```
mkdir build
cd build
cmake ..
make
```

## Test

### Test (C++)
```
cd build
./test_matrixMul
```

### Test (EusLisp)
```
cd euslisp
roseus test-matrixMul.l
(test-matrixMul)
```

## Misc

### Compare matrix multiplication (cuda, cblas, EusLisp)
```
cd euslisp/misc
roseus compare-matrix-multiply.l
(compare-matrix-multiply) ;; cuda is slow only for the first time
(compare-matrix-multiply)
```
