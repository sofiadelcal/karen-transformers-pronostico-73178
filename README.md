# Pronóstico horario con atención y Transformer

Proyecto de Karen Sofía Delgado Calderón para **R2-A3-S12 Transformers (Moodle 73178)**. Compara atención aditiva y un Transformer causal pequeño con persistencia y una referencia estacional de 24 horas. La señal es sintética; no representa mediciones reales.

## Archivos principales

- `Karen_Transformers_Pronostico.ipynb`: cuaderno ejecutable en Google Colab.
- `experimento_transformers.py`: implementación TensorFlow/Keras y guardado de resultados.
- `main.tex`: fuente LaTeX del informe, con formato de estudiante basado en APA 7.
- `Karen_Informe_Transformers_APA.pdf`: informe compilado.
- `resultados/`: métricas observadas en la ejecución, resumen JSON y figuras.

## Ejecución

En Colab, abre el cuaderno y selecciona **Entorno de ejecución → Ejecutar todo**. Localmente:

```bash
python -m pip install -r requirements.txt
python experimento_transformers.py
```

La evaluación usa una ventana de 48 horas, horizonte de seis, corte temporal 80/20 y semillas 7, 23 y 58. En la ejecución reportada (TensorFlow 2.21.0, 33 ventanas de prueba), el RMSE fue 0.6112 para persistencia, 0.3201 para la referencia estacional, 0.6952 ± 0.0031 para atención aditiva y 0.5885 ± 0.0199 para el Transformer. La referencia estacional obtuvo el menor error; las cifras no deben interpretarse como desempeño en datos reales.

Los archivos `metricas_detalladas.csv` y `resumen.json` permiten verificar la tabla del informe.
