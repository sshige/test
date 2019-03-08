# opt_benchmark

## Build
```bash
catkin bt --force-cmake
```
To disable OpenMP,
```bash
catkin bt --force-cmake --cmake-args -DUSE_OPENMP=false
```
To use OpenBLAS,
```bash
catkin bt --force-cmake --cmake-args -DEIGEN3.3_FROM_SOURCE=true -DUSE_OPENBLAS=true
```

## Execute

Solve the nonlienar least square problem with Levenberg-Marquardt method.

```bash
# polynomial problem
# use analytical differentiation
rosrun opt_benchmark eigen_lm # default is --mode default --func poly
# use automatic differentiation
rosrun opt_benchmark eigen_lm --mode ad
# use numerical differentiation
rosrun opt_benchmark eigen_lm --mode nd

# sine problem
# use analytical differentiation
rosrun opt_benchmark eigen_lm --func sin # default is --mode default
# use automatic differentiation
rosrun opt_benchmark eigen_lm --mode ad --func sin
# use numerical differentiation
rosrun opt_benchmark eigen_lm --mode nd --func sin
```
