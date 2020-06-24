#!/bin/bash

ORI="/home/folen/datasets/cuadros/dirs/train"
DST="/home/folen/datasets/cuadros/dirs/valid"

mkdir $DST

for artist in $(ls "$ORI"); do
    mkdir "$DST/$artist"
    for i in $(ls "$ORI/$artist"); do
      # obtengo el número en su nombre
      n=$(echo $i | sed 's/[^0-9]*//g')

      # Si acaba en cero
      if [ $(($n % 13)) -eq "0" ]; then

        # Lo muevo
        mv $ORI/$artist/$i $DST/$artist/$i
      fi
    done
done