# Cómo funcionana las capas embedding

Son muy utilizadas en NLP pero... ¿Cómo funcionan estas capas?

Uso para variables categóricas.

## Datos generales

Vamos a transformar los datos en hombre/Mujer y Bajo, Medio y Alto según su estatura.

La función es la siguiente

```python
def get_data_h():
    g, h, w, i = loadtxt(
        DATA,
        dtype={
            'names': ['Gender', 'Height', 'Weight', 'Index'],
            'formats': ['S1', 'i4', 'i4', 'i4']
        },
        unpack=True,
        skiprows=1,
        delimiter=','
    )

    c = lambda x: 0 if x < 160 else 1 if x < 180 else 2
    h = [c(hs) for hs in h]

    d = np.array([g, h]).T
    return d
```

Si pintamos la información que nos devuelve podemos ver:
```python
[[b'M' b'0']
 [b'M' b'0']
 [b'M' b'1']
 [b'M' b'1']
 [b'F' b'0']
 [b'F' b'2']
 [b'F' b'1']
 [b'F' b'1']
 [b'F' b'0']
 [b'M' b'1']]
```

## Variable categorias con OneHotEncoder

Con estos datos los codificamos con OnHotEncoder y los categorizamos

```python
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(d)

print(enc.categories_)

print(d[0:10])
print(enc.transform(d[0:10]).toarray())
```

Podemos ver la siguiente salida:
```python
[[0. 1. 1. 0. 0.]
 [0. 1. 1. 0. 0.]
 [0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0.]
 [1. 0. 1. 0. 0.]
 [1. 0. 0. 0. 1.]
 [1. 0. 0. 1. 0.]
 [1. 0. 0. 1. 0.]
 [1. 0. 1. 0. 0.]
 [0. 1. 0. 1. 0.]]
```

Donde la columna 0 es si es mujer, la dos si es hombre, la 2 si es bajo, la 3 si es medio y la 4 si es alto.

Esta forma de categorizar guarda información sobre las posible similitudes de los datos de forma discreta (valores 0 o 1 y distancia entre vectores). También tenemos el problema de que si tenemos muchas categorías saldrá un vector muy disperso.

Ejemplo de si combinamos género, altura y peso (4 categorías en peso)
```python
[array([b'F', b'M'], dtype='|S1'), array([b'0', b'1', b'2'], dtype='|S1'), array([b'1', b'2', b'3', b'4'], dtype='|S1')]
[[b'M' b'1' b'3']
 [b'M' b'2' b'3']
 [b'F' b'2' b'4']
 [b'F' b'2' b'4']
 [b'M' b'0' b'2']
 [b'M' b'2' b'4']
 [b'M' b'0' b'3']
 [b'M' b'0' b'4']
 [b'M' b'1' b'3']
 [b'F' b'1' b'4']]
[[0. 1. 0. 1. 0. 0. 0. 1. 0.]
 [0. 1. 0. 0. 1. 0. 0. 1. 0.]
 [1. 0. 0. 0. 1. 0. 0. 0. 1.]
 [1. 0. 0. 0. 1. 0. 0. 0. 1.]
 [0. 1. 1. 0. 0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 1. 0. 0. 0. 1.]
 [0. 1. 1. 0. 0. 0. 0. 1. 0.]
 [0. 1. 1. 0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 1. 0. 0. 0. 1. 0.]
 [1. 0. 0. 1. 0. 0. 0. 0. 1.]]
```

Con información de distancia (solo la distancia de cada vector con el primero)
```python
[[0.         1.         0.         1.         0.         0.         0.         1.         0.         0.        ]
 [0.         1.         0.         0.         1.         0.         0.         1.         0.         1.41421356]
 [1.         0.         0.         0.         1.         0.         0.         0.         1.         2.44948974]
 [1.         0.         0.         0.         1.         0.         0.         0.         1.         2.44948974]
 [0.         1.         1.         0.         0.         0.         1.         0.         0.         2.        ]
 [0.         1.         0.         0.         1.         0.         0.         0.         1.         2.        ]
 [0.         1.         1.         0.         0.         0.         0.         1.         0.         1.41421356]
 [0.         1.         1.         0.         0.         0.         0.         0.         1.         2.        ]
 [0.         1.         0.         1.         0.         0.         0.         1.         0.         0.        ]
 [1.         0.         0.         1.         0.         0.         0.         0.         1.         2.        ]]
```

## Embeddings
