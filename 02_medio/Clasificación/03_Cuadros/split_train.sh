#!/bin/bash

C1="/edgar"
C2="/vicent"

ORI="/home/folen/datasets/cuadros/concrete/train"
DST="/home/folen/datasets/cuadros/concrete/valid"

mkdir $DST
mkdir "$DST/$C1"
mkdir "$DST/$C2"

# Por cada fichero de gatos
for i in $(ls "$ORI/$C1"); do
  # obtengo el número en su nombre
  n=$(echo $i | sed 's/[^0-9]*//g')

  # Si acaba en cero
  if [ $(($n % 13)) -eq "0" ]; then

    # Lo muevo
    mv $ORI/$C1/$i $DST/$C1/$i
  fi
done

# Por cada fichero de gatos
for i in $(ls "$ORI/$C2"); do
  # obtengo el número en su nombre
  n=$(echo $i | sed 's/[^0-9]*//g')

  # Si acaba en cero
  if [ $(($n % 13)) -eq "0" ]; then

    # Lo muevo
    mv $ORI/$C2/$i $DST/$C2/$i
  fi
done