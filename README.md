# CC235-TP_TF_2025-1

---

### Nuestro grupo
Nosotros somos:
* Bruce Maitas Fleck Ojeda - U20211E803
* Alexis Jhon Manuel Davila Mundo - U201922728
* Cristopher Daniel Romero Delgado - U202015120

### Nuestro proyecto
Este proyecto abarca sobre la creacion y despligue de una CNN entrenado para reconocer el 

---

### Dataset
Nuestro dataset fue encontrado por medio de Kaggle llamado ASL Alphabet
El conjunto de datos, que contiene miles de imágenes etiquetadas de gestos manuales que representan cada letra (A-Y, además de clases especiales como «espacio», «del» y «nada»)

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

### Dataset de modelo pre-entrenado
Descargar desde la siguiente dirección:
 https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download

### Modelado
Se preprocesó primero utilizando ImageDataGenerator de TensorFlow. Este paso implicó cambiar el tamaño de las imágenes a 224 × 224 píxeles (para que coincidieran con el tamaño de entrada del modelo), normalizar los valores de los píxeles y aplicar técnicas de aumento de datos como rotación, zoom y desplazamientos para mejorar la capacidad de generalización del modelo. A continuación, el conjunto de datos se dividió en conjuntos de entrenamiento y validación para supervisar el rendimiento del modelo con datos no vistos durante el entrenamiento.

Se crearon 2 modelos para este proyecto, la primera fue creada usando Mobilnet2, un modelo preentrenado que nosotros usamos como referencia para crea nuestro otro modelo original. Nuestro modelo original fue entrenado usando el dataset obtenido de Kaggle a traves de 10 epochs. Y mostro una buena clasificacion de las señas mostradas a la camara.

### Conclusiones
A través de este proyecto, aprendimos a crear un sistema de visión artificial para reconocer el lenguaje de signos americano (ASL) usando aprendizaje automático. Aplicamos aprendizaje por transferencia con MobileNetV2 y entrenamos un clasificador personalizado, logrando alta precisión con un modelo ligero, ideal para aplicaciones en tiempo real. Destacamos la importancia del preprocesamiento, la selección de modelos y la validación, y sentamos las bases para ampliar el sistema a frases completas o gestos dinámicos en el futuro.
