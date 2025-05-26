rm -rf build
mkdir -p build
cd build
CC=gcc-14
CXX=g++-14
cmake .. -DGDT_CUDA_ARCHITECTURES=86 -DOVR_BUILD_MODULE_NNVOLUME=ON -DOVR_BUILD_DEVICE_OSPRAY=OFF -DOVR_BUILD_DEVICE_OPTIX7=OFF
cmake --build . --config Release --parallel 16
ln -s ../data .
cp ../projects/instantvnr/example-model.json .