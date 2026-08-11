<p align="center">
  <img src="assets/icon.png" alt="VideoHighlighter" width="160">
</p>

<!-- hy-mt2-i18n:start -->
[English](./README.md) | [中文](./README_zh-CN.md) | [日本語](./README_ja.md) | **Español**
<!-- hy-mt2-i18n:end -->

VideoHighlighter (Software gratuito)

Herramienta en Python para generar automáticamente clips destacados a partir de videos, mediante detección de escenas, detección de movimiento, picos de audio, detección de objetos, reconocimiento de acciones y análisis de transcripciones.

> **Es gratis.** Para no perderte las próximas versiones, pulsa el botón de
> motivación: la ⭐ de arriba. Es el pago más barato que aceptamos.


Funciones

Detección:
- Escenas mediante OpenCV.
- Picos de movimiento y cambios de escena.
- Objetos
- Acciones
- Picos de audio.

Genera subtítulos a partir de la transcripción mediante OpenAI Whisper.  
Recorta y combina los segmentos con mayor puntuación para crear un video de resúmenes.  
Totalmente configurable: salto de fotogramas, duración de los resúmenes y palabras clave.  
Incluye una interfaz gráfica opcional para una interacción sencilla.

¿No está seguro de qué detector utilizar? Consulte
[docs/DETECTION-GUIDE.md](docs/DETECTION-GUIDE.md): qué son capaces de hacer el reconocimiento de objetos, el reconocimiento de acciones, la búsqueda CLIP y el motor de composición, así como sus limitaciones.

> **¿Desea detección en tiempo real?** Todo lo anterior funciona sin conexión, posteriormente a la reproducción.
> [VideoHighlighter Pro](#pro-edition) agrega superposiciones en tiempo real de objetos y acciones
> durante la reproducción, permite categorías por ejemplo, detección con vocabulario abierto y detección de contadores. [Vea las diferencias →](#pro-edition)


## Vista previa

![VideoHighlighter](assets/Highlighter.png)

## Visor de línea de tiempo
![Timeline Viewer](assets/TimelineViewer.png)

## Reconocimiento de acciones
![Reconocimiento de acciones](assets/power_rangers_actions_annotated.gif)

## Etapas del flujo de trabajo
![Workflow Stages](assets/workflow_stages.png)

## Edición Pro

Esta edición ya incluye detección facial en tiempo real, reproducción y renderizado en VR lado a lado, análisis sin conexión, búsqueda con CLIP, el motor de composición y los scripts de entrenamiento.

[VideoHighlighter Pro](https://aseiel.github.io/VideoHighlighter-site/) agrega:

- **Superposiciones en tiempo real de objetos y acciones**: detección en vivo durante la reproducción, incluso en grabaciones VR lado a lado.  
- **Enseñar una categoría señalando**: dibuja un cuadro alrededor de cualquier elemento, dale un nombre, y se le asignará una puntuación en tiempo real a partir de ese momento. Sin necesidad de conjunto de datos ni entrenamiento.  
- **Encontrar elementos similares**: elige una región en una imagen y busca toda la videocámara por ella.  
- **Detección con vocabulario abierto**: escribe una palabra cualquiera y úsala para realizar búsquedas, sin necesidad de un modelo entrenado previamente.  
- **Detección de contadores/puntuaciones**: si la grabación muestra un contador en pantalla, cada marcador representa un evento; así, la versión Pro puede indicar qué momentos reales pasaron desapercibidos para el detector.

Esta versión sigue siendo gratuita y está licenciada bajo AGPL-3.0.

## Instalación

### Windows (recomendado)
Descargue la última versión en formato `.exe` desde [Releases](https://github.com/Aseiel/VideoHighlighter/releases); no se requiere Python ni otras dependencias.

### Linux / Compilación desde el código fuente
1. **Python & FFmpeg**
   Es necesario tener instalado FFmpeg y que esté disponible en el PATH de su sistema.

## Uso
Linux: python main.py 
Windows: ejecutar Videohighlighter.exe
Mac: Creo que no funciona, lo arreglaré algún día. Todavía se genera el archivo DMG

## Discord
De vez en cuando, VideoHighlighter “siente” algo al respecto de tus grabaciones. Cuando eso ocurre:
[Únete a Discord](https://discord.gg/cUPJqPAMmm) y escribe en #support; por lo general estoy allí.


## Notas

OpenAI Whisper está bajo licencia MIT, por lo que se puede utilizar libremente.

La API de Google Translate es opcional. Si se utilizan bibliotecas no oficiales (como googletrans), no se necesita clave de API, pero los resultados podrían dejar de funcionar si Google modifica sus puntos de conexión.

Este proyecto no incluye claves API de pago. Los usuarios deben proporcionar las suyas propias si utilizan servicios oficiales.


## Licencia

Este repositorio se publica bajo la licencia GNU Affero General Public License v3.0 (AGPLv3). Puede utilizar, modificar y distribuir el código libremente, siempre y cuando cualquier versión modificada, incluidas las ofrecidas a través de una red, haga disponible su código fuente completo bajo la misma licencia.


## Antecedentes del proyecto

Este proyecto nació como una herramienta personal para generar automáticamente subtítulos de videos, destinada a mi hijo de 7 años. Con el tiempo, se transformó en un generador de resúmenes para películas, eventos deportivos y videos personales.

El objetivo principal sigue siendo práctico: acelerar el análisis de videos, generar resúmenes destacados y crear subtítulos accesibles de forma automática.

![Historial de estrellas](assets/star-history-2026630.png)
