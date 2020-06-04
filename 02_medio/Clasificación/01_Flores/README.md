# Clasificación de flores

Dentro de los posibles dataset que existen en internet hay uno de 102 clases con imágenes de flores. La información está en este [enlace](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

Estos dataset de pruebas y formación suelen estar incluidos en alguna librería. A Día de hoy (04/06/2020) está contenido en ```tfds-nightly```.

## Cargando dataset tensorflow

Cargamos el dataset y una información de los datos

```python

import tensorflow_datasets as tfds

# LA primera vez descargará los datos
labeled_ds, summary = tfds.load('oxford_flowers102', split='train+test+validation', with_info=True)

```

LA información es la siguiente

```python

tfds.core.DatasetInfo(
    name='oxford_flowers102',
    version=2.1.1,
    description='The Oxford Flowers 102 dataset is a consistent of 102 flower categories commonly occurring
in the United Kingdom. Each class consists of between 40 and 258 images. The images have
large scale, pose and light variations. In addition, there are categories that have large
variations within the category and several very similar categories.

The dataset is divided into a training set, a validation set and a test set.
The training set and validation set each consist of 10 images per class (totalling 1020 images each).
The test set consists of the remaining 6149 images (minimum 20 per class).',
    homepage='https://www.robots.ox.ac.uk/~vgg/data/flowers/102/',
    features=FeaturesDict({
        'file_name': Text(shape=(), dtype=tf.string),
        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
        'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=102),
    }),
    total_num_examples=8189,
    splits={
        'test': 6149,
        'train': 1020,
        'validation': 1020,
    },
    supervised_keys=('image', 'label'),
    citation="""@InProceedings{Nilsback08,
       author = "Nilsback, M-E. and Zisserman, A.",
       title = "Automated Flower Classification over a Large Number of Classes",
       booktitle = "Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing",
       year = "2008",
       month = "Dec"
    }""",
    redistribution_info=,
)


```