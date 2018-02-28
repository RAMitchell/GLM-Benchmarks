set -e
rm -rf xgboost
git clone https://github.com/RAMitchell/xgboost.git --recursive
cd xgboost
#cmake
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DUSE_AVX=ON
make -j4
cd ..
cd python-package/
python setup.py install

