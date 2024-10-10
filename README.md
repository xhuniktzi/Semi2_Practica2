
# Análisis de Datos con Python - Práctica 2

## Universidad de San Carlos de Guatemala

**Facultad de Ingeniería**  
**Escuela de Ciencias y Sistemas**  
**Seminario de Sistemas 2**  
**Sección N**  
**Ing. Fernando Paz**  
**Aux. Sergio Enrique Cubur Chalí**  

---

### Objetivos

1. Desarrollar un notebook para el análisis de datos que permita la carga, manipulación, visualización y generación de informes utilizando Pandas, NumPy y Matplotlib.
2. Implementar técnicas de limpieza y transformación de datos para preparar los datos para el análisis.
3. Crear visualizaciones interactivas que permitan explorar visualmente los datos para la toma de decisiones.

---

## Paso 1: Importación de Librerías

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Paso 2: Carga de Datos

Se carga el archivo `CSV` proporcionado y se almacena en un DataFrame.

```python
df = pd.read_csv('data.csv')
```

---

## Paso 3: Limpieza de Datos

Se realiza la limpieza de datos para asegurar que estén en el formato correcto y sean adecuados para el análisis.

- Eliminar duplicados:

  ```python
  df.drop_duplicates(inplace=True)
  ```

- Verificar valores faltantes:

  ```python
  print("Datos faltantes por columna:")
  print(df.isnull().sum())
  ```

- Rellenar o eliminar valores faltantes:

  ```python
  df.dropna(subset=['Rating', 'Level', 'Duration', 'Schedule', 'Review', 'What you will learn'], inplace=True)
  ```

---

## Paso 4: Cálculos Requeridos

### Promedio de Calificaciones por Curso

Se calcula el promedio de las calificaciones para cada curso.

```python
average_rating = df.groupby('Course Title')['Rating'].mean().reset_index()
```

### Curso con Mayor y Menor Calificación

Identificación de los cursos con el mayor y menor rating.

```python
highest_rated_course = df.loc[df['Rating'].idxmax()]
lowest_rated_course = df.loc[df['Rating'].idxmin()]
```

### Porcentaje de Cursos con Horario Flexible

Se calcula el porcentaje de cursos que tienen un horario flexible en relación con el total de cursos.

```python
flexible_courses = df[df['Schedule'].str.contains('flexible', case=False, na=False)]
flexible_percentage = round((len(flexible_courses) / len(df)) * 100, 2)
```

---

## Paso 5: Análisis de Texto (NLTK)

### Tokenización

Se convierte el texto en una lista de palabras (tokens).

```python
tokens = word_tokenize(texto.lower())
```

### Lematización y Stemming

Se aplica lematización y stemming para normalizar las palabras.

```python
lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
stemmed_words = [stemmer.stem(token) for token in tokens]
```

### Eliminación de Palabras Vacías

Se eliminan las palabras comunes sin significado importante (stopwords).

```python
filtered_words = [word for word in lemmatized_words if word.isalnum() and word not in stop_words]
```

### Frecuencia de Palabras

Se genera una distribución de frecuencias para las palabras.

```python
freq_dist = FreqDist(filtered_words)
```

### Análisis de Sentimientos

Se realiza un análisis de sentimientos sobre el texto.

```python
sentimientos = sia.polarity_scores(texto)
```

### Reconocimiento de Entidades Nombradas

Se extraen entidades nombradas del texto.

```python
entidades = ne_chunk(pos_tag(word_tokenize(texto)))
```

---

## Paso 6: Visualizaciones

### Gráfica de Barras

Número de cursos por nivel de dificultad.

```python
df['Level'].value_counts().plot(kind='bar')
plt.title('Número de cursos por nivel')
plt.xlabel('Nivel')
plt.ylabel('Número de cursos')
plt.show()
```

### Gráfica de Barras Horizontal

Número de cursos en las principales categorías.

```python
df['Category'].value_counts().plot(kind='barh')
plt.title('Número de cursos por categoría')
plt.xlabel('Número de cursos')
plt.ylabel('Categoría')
plt.show()
```

### Gráfico de Dispersión

Relación entre la duración del curso y el número de revisiones.

```python
plt.scatter(df['Duration'], df['Review'])
plt.title('Relación entre duración y número de revisiones')
plt.xlabel('Duración (horas)')
plt.ylabel('Número de revisiones')
plt.show()
```

### Histograma

Distribución de las duraciones de los cursos.

```python
df['Duration'].dropna().plot(kind='hist', bins=20)
plt.title('Distribución de la duración de los cursos')
plt.xlabel('Duración (horas)')
plt.ylabel('Frecuencia')
plt.show()
```

### Gráfico de Cajas

Distribución de las calificaciones de los cursos por nivel.

```python
df.boxplot(column='Rating', by='Level')
plt.title('Distribución de calificaciones por nivel')
plt.xlabel('Nivel')
plt.ylabel('Calificación')
plt.show()
```

---
Para los puntos del paso 7 (Análisis y Resultados) y las conclusiones, podemos basarnos en la información proporcionada:

### Paso 7: Análisis y Resultados

1. **Análisis del Promedio de Calificaciones por Curso:**
   - El cálculo muestra una variedad de calificaciones entre los cursos, con algunos alcanzando un rating perfecto de 5.0, mientras que otros presentan calificaciones bajas, alrededor de 3.0. Esto indica una amplia gama de calidad percibida por los usuarios.

2. **Cursos con Mayor y Menor Calificación:**
   - El curso con la calificación más alta es "El Holocausto: el abismo de la humanidad" con un rating de 5.0, lo que indica que los usuarios encuentran este curso muy valioso y bien estructurado.
   - El curso con la calificación más baja es "Software Architecture for Big Data Specialization," con un rating de 3.1. Es posible que este curso enfrente desafíos en términos de contenido o presentación que afectan su percepción.

3. **Porcentaje de Cursos con Horario Flexible:**
   - Un alto porcentaje de cursos tienen horarios flexibles, lo cual es un factor importante para los estudiantes que buscan adaptabilidad en sus estudios. Esto refleja una tendencia en las plataformas de aprendizaje en línea para ofrecer opciones de aprendizaje más accesibles.

4. **Análisis de Texto con NLTK:**
   - La tokenización y lematización del texto han permitido identificar las palabras más comunes en el archivo de texto, con "semana," "curso," y "participantes" apareciendo con alta frecuencia.
   - El análisis de sentimientos muestra que la mayoría de los comentarios son neutros, lo que sugiere que el tono del archivo de texto es principalmente informativo en lugar de emocional.
   - El reconocimiento de entidades nombradas no proporcionó muchas entidades específicas, lo que puede indicar que el archivo de texto no estaba orientado a nombrar individuos u organizaciones.

### Conclusiones

1. **Sobre el Análisis de Datos:**
   - El uso de técnicas de limpieza de datos fue crucial para asegurar que los análisis fueran precisos y útiles. La eliminación de duplicados y la gestión de valores faltantes mejoraron significativamente la calidad de los datos.
   - Los gráficos generados proporcionaron información valiosa sobre la distribución de los cursos, la duración y las calificaciones, permitiendo identificar tendencias en los datos.

2. **Sobre el Uso de Python:**
   - Python, con librerías como Pandas, NumPy, Matplotlib y NLTK, demostró ser una herramienta poderosa para el análisis de datos y la visualización. La capacidad de combinar diferentes tipos de análisis (numéricos y de texto) en un solo entorno facilitó el proceso.
   - La facilidad para generar visualizaciones y análisis textuales con Python ayuda a presentar los datos de manera clara y efectiva, lo cual es fundamental para la toma de decisiones basada en datos.

3. **Recomendaciones:**
   - Para mejorar la percepción de algunos cursos, se podría considerar la reestructuración del contenido o mejorar la presentación de los cursos con calificaciones bajas.
   - Dado el alto porcentaje de cursos con horarios flexibles, la plataforma podría destacar estas opciones como un punto de venta clave para atraer más estudiantes.
