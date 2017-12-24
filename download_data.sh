#!/bin/bash 
mkdir -p data
cd data
data_files="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
  http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
  http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
  http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
for f in $data_files
do
  wget $f
done
cd -
