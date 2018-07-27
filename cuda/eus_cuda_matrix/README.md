## Document
matrixMul.cu in
https://github.com/NVIDIA/cuda-samples/tree/master/Samples/matrixMul
is edited and used.

Number of columns and rows of matrix should be multiples of 32.

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
