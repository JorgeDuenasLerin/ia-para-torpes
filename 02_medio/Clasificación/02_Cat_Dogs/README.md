# Juego con el dataset Cat and Dogs

Trabajaremos en esta sesión con imágenes de perros y gatos dentro del dataset:
[Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)

NOTA: Descargar datos una vez que hemos aceptado términos para conseguir los datos en dos conjuntos (train-test)

Índice:
- Jugando con los datos. Este README.md
- [Mi propia red](REDPROPIA.md)
- [Qué hay dentro de la red](FILTROS.md)
- [Transferencia de aprendizaje](TRANSFERLEARNING.md)

## Análisis de la información

Vamos a observar qué información tenemos y cómo la tenemos.

El directorio tiene la siguiente estructura:
```python
dogs-vs-cats/
 \- train
    \- cat.NN.jpg
    \- ...
    \- dog.NN.jpg
    \- ...
 \- test1
    \- NNNNN.jpg
    \- NNNNN.jpg
    \- NNNNN.jpg
    \- NNNNN.jpg
    \- NNNNN.jpg
    \- ...
```

¿Qué resolución tienen?

```shell script
>file train/cat/cat.1111* | cut -d  "," -f8
 499x333
 500x374
 499x375
 179x186
 499x405
 249x222
 500x374
 375x499
 500x374
 500x332
 499x375
```

Vemos que tienen resoluciones distintas con lo que habrá que redimensionar, también vemos que no están colocadas como espera ```ImageDataGenerator```. necesitamos tener la información clasificada en carpetas, así que movemos los gatos a la carpeta gatos y los perror a la carpeta perros.

```shell script
[folen:~/dogs-vs-cats/train] $ 
>mkdir cat
[folen:~/dogs-vs-cats/train] $ 
>mv cat.* cat
[folen:~/dogs-vs-cats/train] $ 
>mkdir dog
[folen:~/dogs-vs-cats/train] $ 
>mv dog.* dog
[folen:~/dogs-vs-cats/train] $
```

¿Qué tenemos en el conjnto de test?
```shell script
>ls test1/
10000.jpg  10705.jpg  11409.jpg  12112.jpg  1566.jpg  226.jpg   2974.jpg  
10001.jpg  10706.jpg  1140.jpg   12113.jpg  1567.jpg  2270.jpg  2975.jpg  
10002.jpg  10707.jpg  11410.jpg  12114.jpg  1568.jpg  2271.jpg  2976.jpg  
10003.jpg  10708.jpg  11411.jpg  12115.jpg  1569.jpg  2272.jpg  2977.jpg  
10004.jpg  10709.jpg  11412.jpg  12116.jpg  156.jpg   2273.jpg  2978.jpg  
10005.jpg  1070.jpg   11413.jpg  12117.jpg  1570.jpg  2274.jpg  2979.jpg  
10006.jpg  10710.jpg  11414.jpg  12118.jpg  1571.jpg  2275.jpg  297.jpg   
10007.jpg  10711.jpg  11415.jpg  12119.jpg  1572.jpg  2276.jpg  2980.jpg  
10008.jpg  10712.jpg  11416.jpg  1211.jpg   1573.jpg  2277.jpg  2981.jpg  
...  
...
```

Tenemos muchas imágenes sin etiquetar!

El trabajo que debemos hacer es dividir el train en train y validation. Una vez entrenada y validada probar con el conjunto de test, del que no sabemos sus etiquetas, y con un csv y su clasificación comprobar el grado de exactitud. 

## Sacamos el conjunto de validation

Para tener conjuntos valanceados voy a mover cada fichero cuyo id sea divisible por 13 de los gatos y luego de los perros

```shell script
1/13 = 0.076 
```

Así tendremos un 7% de los datos para validación

El código se encuentra en el fichero ``` split_train.sh ```. El trozo importante está en:

```shell script
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
```

Par comprobar que tenemos dos conjuntos utilizamos algunos comandos bash:

```shell script
[folen:~/dogs-vs-cats] $ 
>ll train/cat/ | wc
  11541  103862  682136
[folen:~/dogs-vs-cats] $ 
>ll valid/cat/ | wc
    965    8678   56009
[folen:~/dogs-vs-cats] $ 
>ll train/dog/ | wc
  11541  103862  682136
[folen:~/dogs-vs-cats] $ 
>ll valid/dog/ | wc
    965    8678   56009
```

## Comprobamos etiqueta

Los gatos debe ser 0 y los perros 1. La clasificación la hace por el orden alfabético de la carpeta, aún así hay un código comentado para mostrar una imágen y su etiqueta y es así. Tambień podemos ver como el ImageGenerator baraja la información y cada vez nos enseña una foto.

```python
# Pinta imágenes y comprueba etiqueta
e=next(iter(train_generator))

array_image = e[0][0]
value_label = e[1][0]

plt.imshow(array_image, interpolation='nearest')
plt.title(value_label)
plt.show()
``` 

## Continuar

[Entrenando mi propia red](REDPROPIA.md)