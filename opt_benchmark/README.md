# opt_benchmark

## Build
```bash
catkin bt --force-cmake # Same with the following options if this is first catkin build. Otherwise, specify options explicitly.
# catkin bt --force-cmake --cmake-args -DEIGEN3.3_FROM_SOURCE=false -DUSE_OPENBLAS=false -DUSE_OPENMP=true
```
To disable OpenMP,
```bash
catkin bt --force-cmake --cmake-args -DUSE_OPENMP=false
```
To use OpenBLAS,
```bash
catkin bt --force-cmake --cmake-args -DEIGEN3.3_FROM_SOURCE=true -DUSE_OPENBLAS=true
```

### Use latest eigen

For using latest eigen (= make `EIGEN3.3_FROM_SOURCE` true),
```
mkdir ~/src ~/install
cd ~/src
hg clone https://bitbucket.org/eigen/eigen/
cd eigen
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/install/
make
make install
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
