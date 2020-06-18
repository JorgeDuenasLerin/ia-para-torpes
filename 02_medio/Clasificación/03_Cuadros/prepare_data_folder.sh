#!/bin/bash

DIR=/home/folen/datasets/cuadros/resized
DIR_OUT=/home/folen/datasets/cuadros/dirs

mkdir $DIR_OUT

for f in $DIR/*; do
  file=${f##*/}
  folder=${file%_*}
  mkdir $DIR_OUT/$folder
  cp $f $DIR_OUT/$folder
done