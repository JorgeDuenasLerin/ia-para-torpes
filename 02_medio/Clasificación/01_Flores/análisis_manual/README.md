# Clasificación de flores

Dentro de los posibles dataset que existen en internet hay uno de 102 clases con imágenes de flores. La información está en este [enlace]

http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

Probando la clasificación de flores

http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt

## Analizando la información

Existe un fichero ``` setid.mat ```. Dentro del fichero ``` probando_datos.py ``` está el código para mostrar la información.

Tras la ejecución podemos ver:
```python
{   
    '__globals__': [],
    '__header__': b'MATLAB 5.0 MAT-file, Platform: GLNX86, Created on: Thu F'
                  b'eb 19 17:38:58 2009',
    '__version__': '1.0',
    'trnid': array([[6765, 6755, 6768, ..., 8026, 8036, 8041]], dtype=uint16),
    'tstid': array([[6734, 6735, 6737, ..., 8044, 8045, 8047]], dtype=uint16),
    'valid': array([[6773, 6767, 6739, ..., 8028, 8008, 8030]], dtype=uint16)
}
```

Parece que tenemos
- trnid: entrenamiento
- tsnid: test
- valid: validacion

Si ejecutamos un len sobre los arrays vemos que contienen el siguiente número de ids
```python
>>> dict_mat_contents['trnid']
array([[6765, 6755, 6768, ..., 8026, 8036, 8041]], dtype=uint16)
>>> len(dict_mat_contents['trnid'])
1
>>> len(dict_mat_contents['trnid'][0])
1020
>>> len(dict_mat_contents['tstid'][0])
6149
>>> len(dict_mat_contents['valid'][0])
1020
>>> 

```

## Información

Parece que existe este conjunto de datos dentro de ```tfds-nightly```



 