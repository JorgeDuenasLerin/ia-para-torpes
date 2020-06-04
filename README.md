# IA para torpes
Comenzado a investigar ML y DL

## Entorno
Descripción del entorno necesario... vieniendo del desarrollo


> anaconda

Plataforma opensource para gestionar aplicaciones, librerías 

Ejecución tras [instalar](https://www.anaconda.com/products/individual)

```bash
~/anaconda3/bin/anaconda-navigator
```

> conda

Agrupa la funcionalidad de gestor de paquetes (pip o npm) y de entorno virtuales (virtualenv). Los paquetes soportados por el gestor de paquetes son multi lenguaje: Python, R, Ruby, Lua, Scala, Java, JavaScript, C/ C++, FORTRAN

En la instalación por defecto cra un virtualenv que se activa. Molesto si tienes proyectos con otros virtualenvs. Para desactivarlo.

```bash
conda config --set auto_activate_base false
```  

Para activarlo cuando quiera usarlo

```bash
conda activate
conda deactivate
```

En realidad es un virtualenv en ``` ~/anaconda3/bin ```. Útil cuando lo integres en el interprete del IDE.

## Listado de ejemplos

1. Básicos
    1. [Puerta XOR](01_básicos/01_XOR/)