#!/bin/bash

CAT="/cat"
DOG="/dog"

ORI="/home/folen/dogs-vs-cats/train"
DST="/home/folen/dogs-vs-cats/valid"

mkdir $DST
mkdir "$DST/$CAT"
mkdir "$DST/$DOG"

# Por cada fichero de gatos
for i in $(ls "$ORI/$CAT"); do
  # obtengo el número en su nombre
  n=$(echo $i | sed 's/[^0-9]*//g')

  # Si acaba en cero
  if [ $(($n % 13)) -eq "0" ]; then

    # Lo muevo
    mv $ORI/$CAT/$i $DST/$CAT/$i
  fi
done

# Por cada fichero de gatos
for i in $(ls "$ORI/$DOG"); do
  # obtengo el número en su nombre
  n=$(echo $i | sed 's/[^0-9]*//g')

  # Si acaba en cero
  if [ $(($n % 13)) -eq "0" ]; then

    # Lo muevo
    mv $ORI/$DOG/$i $DST/$DOG/$i
  fi
done