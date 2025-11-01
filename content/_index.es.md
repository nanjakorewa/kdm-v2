---
title: "K_DM Book | Guía integral de análisis de datos y aprendizaje automático"
linkTitle: "Inicio"
seo_title: "K_DM Book | Guía integral de análisis de datos y aprendizaje automático"
---
## Bienvenido a K_DM Book
K_DM Book es una guía práctica que te acompaña desde los fundamentos del análisis de datos hasta la puesta en producción de modelos. Cada sección combina explicaciones con notebooks y scripts ejecutables para que puedas aprender los conceptos y aplicarlos de inmediato.

- **Ruta de aprendizaje estructurada** — La estructura permite avanzar sin lagunas desde los conceptos básicos hasta los flujos productivos.
- **Ejemplos reproducibles** — La mayoría de los artículos incluyen notebooks o scripts de Python que generan las figuras y resultados.
- **Consejos orientados al trabajo real** — Documentamos métricas de evaluación, listas de verificación de preprocesamiento y patrones de visualización basados en proyectos reales.

## Itinerario recomendado
1. **Fundamentos** — Repasa modelos clave como regresión, clasificación, agrupamiento y reducción de dimensionalidad.
2. **Preparación / Evaluación** — Profundiza en ingeniería de características y validación de modelos para iterar con confianza.
3. **Timeseries / Finance / WebApp** — Explora casos de uso específicos y aprende cómo desplegar soluciones.
4. **Visualize** — Mejora la comunicación de resultados a través de gráficos, dashboards y reportes eficaces.

## Mapa de contenidos
{{< mermaid >}}
flowchart TD
  subgraph fundamentals["Fase de Fundamentos"]
    A1[Bases matemáticas<br>Álgebra lineal / Probabilidad]
    A2[Configuración del entorno<br>Python y librerías]
    A3[Análisis exploratorio<br>EDA y visualización básica]
  end

  subgraph modeling["Fase de Modelado Básico"]
    B1[Regresión]
    B2[Clasificación]
    B3[Agrupamiento]
    B4[Reducción de dimensionalidad]
    B5[Métodos en conjunto]
  end

  subgraph evaluation["Fase de Evaluación y Ajuste"]
    C1[Métricas]
    C2[Ajuste de hiperparámetros]
    C3[Ingeniería de características]
    C4[Interpretabilidad del modelo]
  end

  subgraph communication["Fase de Comunicación"]
    D1[Catálogo de visualizaciones]
    D2[Narrativas y reportes]
    D3[Consejos de colaboración]
  end

  subgraph deployment["Fase de Aplicación y Despliegue"]
    E1[Análisis de series temporales]
    E2[Casos financieros]
    E3[Entrega web<br>Flask / Gradio]
    E4[Monitoreo y operaciones]
  end

  fundamentals --> modeling
  modeling --> evaluation
  evaluation --> communication
  evaluation --> deployment
  communication --> deployment

  click A2 "/install/" "Ir a la guía de instalación"
  click A3 "/prep/" "Abrir la sección de preparación de datos"
  click B1 "/basic/regression/" "Ir a regresión"
  click B2 "/basic/classification/" "Ir a clasificación"
  click B3 "/basic/clustering/" "Ir a agrupamiento"
  click B4 "/basic/dimensionality_reduction/" "Ir a reducción de dimensionalidad"
  click B5 "/basic/ensemble/" "Ir a métodos en conjunto"
  click C1 "/eval/" "Ver métricas de evaluación"
  click C3 "/prep/feature_selection/" "Ir a ingeniería de características"
  click D1 "/visualize/" "Explorar la sección Visualize"
  click E1 "/timeseries/" "Abrir contenidos de series temporales"
  click E2 "/finance/" "Abrir contenidos de finanzas"
  click E3 "/webapp/" "Abrir contenidos de aplicaciones web"
{{< /mermaid >}}

## Recursos y soporte
- **Notebooks y scripts** — Consulta `scripts/` y `data/` para reproducir los tutoriales.
- **Novedades** — Sigue la cronología de la portada o suscríbete al feed RSS (`/index.xml`) para recibir actualizaciones.
- **Comentarios** — ¿Detectaste un error? Envíanos tus sugerencias mediante el [formulario de contacto](https://kdm.hatenablog.jp/entry/issue) o a través de X / Twitter.

## Redes y comunidad
- <a href="https://www.youtube.com/@K_DM" style="color:#FF0000;"><i class="fab fa-fw fa-youtube"></i> YouTube — vídeos y sesiones en directo</a>
- <a href="https://twitter.com/_K_DM" style="color:#1DA1F2;"><i class="fab fa-fw fa-twitter"></i> X (Twitter) — novedades rápidas y consejos breves</a>

Para más detalles revisa la [política de privacidad](https://kdm.hatenablog.jp/privacy-policy). ¡Disfruta el material y adáptalo a tus proyectos!
