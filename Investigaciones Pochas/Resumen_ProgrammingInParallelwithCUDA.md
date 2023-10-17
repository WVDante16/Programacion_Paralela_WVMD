Programming in Parallel with CUDA
Richard Ansorge

Background.
Un procesador de PC moderno ahora tiene dos, cuatro o más núcleos de CPU cores. Para obtener lo mejor desde dicho hardware, su código debe poder ejecutarse en paralelo en todos los recursos disponibles. En casos favorables, herramientas como OpenMP o la clase de subproceso C++ 11 definida en <thread> le permite lanzar subprocesos cooperativos en cada uno de los núcleos de hardware para obtener un potencial de aceleración proporcional al número de núcleos.
Una alternativa es equipar su PC con una GPU moderna y con especificaciones razonablemente altas.
En el corazón de cualquier programa CUDA hay una o más funciones del núcleo, que contiene el código que realmente se ejecuta en la GPU. Estas funciones están escritas en C++ estándar con un pequeño número de extensiones y restricciones.
Para obtener lo mejor de los programas CUDA (y de hecho de cualquier otro lenguaje de programación), es necesario tener una comprensión básica del hardware subyacente.

CPU Architecture.
Se puede escribir un código informático correcto simplemente siguiendo las reglas formales del sistema en particular del lenguaje que se utiliza. Sin embargo el código compilado en realidad se ejecuta en hardware físico, por lo que es útil tener algunas ideas sobre las limitaciones del hardware al diseñar sistemas de código de alto rendimiento.
Brevemente los bloques que se muestran son:
Master Clock: El reloj actúa como el director de una orquesta, pero suena una melodía muy aburrida. Se envían pulsos de reloj a una frecuencia fija a cada unidad, lo que hace que esa unidad ejecute su próximo paso. La velocidad de procesamiento de la CPU es directamente proporcional a esta frecuencia.
Memory: La memoria principal contiene tanto los datos del programa como las instrucciones del código de máquina generadas por el compilador a partir de su código de alto nivel.
Load/Save: Esta unidad lee datos y envía datos a la memoria principal. La unidad está controlada por la lógica de ejecución que en cada paso específica si los datos se leerán o escribirán en la memoria principal.
Register File: Este es el corazón de una CPU; Los datos deben almacenarse en uno o más registros en orden para ser operado por la ALU.
ALU o Arithmetic Logic Unit: Este dispositivo realiza operaciones aritméticas y lógicas en datos almacenados en registros; en cada paso la operación requerida es especificada por la unidad de ejecución.
Execute: Esta unidad decodifica la instrucción enviada por la unidad de búsqueda de instrucciones, organiza la transferencia de datos de entrada al archivo de registro, indica a ALU que realice la operación requerida sobre los datos y finalmente organiza la transferencia de los resultados a la memoria.
Fetch: La unidad de recuperación recupera instrucciones de la memoria principal y las pasa a la unidad de ejecución. La unidad contiene un registro que contiene el contador de programa (PC) que contiene la dirección de la instrucción actual. La PC normalmente se incrementa en una instrucción en cada paso para que las instrucciones se ejecuten secuencialmente.

CPU Compute Power.
La potencia informática de las CPU ha aumentado espectacularmente con el tiempo.
El número de transistores por chip ha seguido el crecimiento exponencial de la ley de Moore desde 1970. 
La frecuencia de la CPU dejó de aumentar en 2002. El rendimiento por núcleo continúa aumentando lentamente, pero el crecimiento reciente se debe a innovaciones en el diseño más que a un aumento en la frecuencia. La mayor contribución al rendimiento por chip desde 2002 ha sido de tecnología multinúcleo.
La potencia informática de los dispositivos individuales se ha multiplicado por más de 106 en los últimos 30 años y además el número de dispositivos ha crecido a un ritmo aún más rápido. Mucha gente ahora posee numerosos dispositivos en sus teléfonos inteligentes, computadoras portátiles, automóviles y aparatos domésticos, mientras que los gigantes de internet administran enormes granjas de servidores, cada una de las cuales debe tener millones de procesadores en funcionamiento.

CPU Memory Management: Latency Hiding Using Caches.
Los datos y las instrucciones no se mueven instantáneamente entre los bloques; bastante avanzan pero paso a paso a través de varios registros de hardware desde la fuente hasta el destino de modo que haya una latencia entre la emisión de una solicitud de datos y la llegada de esos datos. Esta latencia intrínseca suele ser de decenas de ciclos de reloj en una CPU y cientos de ciclos de reloj en una GPU. Afortunadamente, las implicaciones potencialmente desastrosas para el rendimiento de la latencia se pueden ocultar en gran medida mediante una combinación de almacenamiento en caché y canalización.
En la práctica, las PC emplean varias unidades de memoria caché para almacenar en buffer la transmisión de datos desde múltiples lugares en la memoria principal.
Las instrucciones del programa también se transmiten en cadena desde la memoria principal a la unidad de búsqueda de instrucciones. Esta canalización se interrumpe si se produce una instrucción de bifurcación y el hardware de la PC utiliza trucos sofisticados como la ejecución especulativa para minimizar los efectos.
El esquema de almacenamiento en caché es típico de los chips multinúcleo de CPU modernos. Hay tres niveles de memoria caché, todas integradas en el chip de la CPU. La caché (L3) es grande, de 8 MB, y la comparten los 4 núcleos de la CPU. Cada núcleo también tiene caches L2 y L1 progresivamente más rápidos, con cachés L1 separados para datos e instrucciones.
El hardware transfiere datos de caché en paquetes llamados líneas de caché de normalmente 64 o 128 bytes. En los procesadores Intel actuales, el tamaño de la línea de caché es de 64 bytes, pero el hardware generalmente transfiere dos líneas adyacentes a la vez dando un tamaño efectivo de 128 bytes.

CPU: Parallel Instruction Set.
Antes de analizar las poderosas capacidades paralelas de las GPU, vale la pena señalar que las CPU de Intel también tienen algunas capacidades paralelas interesantes en forma de instrucciones vectoriales. Estos aparecieron por primera vez alrededor de 1999 con el conjunto de instrucciones Pentium III SSE. Estas instrucciones utilizaron ocho nuevos registros de 128 bits, cada uno de ellos capaz de contener 4 flotantes de 4 bytes. Si conjuntos de 4 de estos flotantes se almacenarán en ubicaciones de memoria secuenciales y se alinearon en límites de memoria de 128 bytes, se pueden tratar como un vector de 4 elementos y estos vectores se pueden cargar y almacenar desde los registros SSE en un solo ciclo de reloj y las ALU mejoradas también podrían realizar aritmética vectorial en ciclos de reloj individuales.

GPU Architecture.
Las GPU se diseñaron para gráficos de computadora de alto rendimiento; en los juegos modernos, un tamaño de pantalla de 1920 x 1080 pixeles actualizado a 60 Hz es normal. Cada pixel debe ser calculado a partir del estado instantáneo del juego y en el punto de vista del jugador aproximadamente 1,25 cálculos de 108 píxeles por segundo. Las tarjetas de juego surgieron como hardware dedicado con una gran cantidad de procesadores simples para realizar los cálculos de píxeles. Un detalle técnico importante es que la matriz de píxeles que representa la imagen se almacena en una matriz 2D en un buffer de fotograma digital como datos, normalmente 3 bytes por pixel que representan los colores rojo, verde y azul (RGB), como intensidades del pixel.
Rápidamente se notó que una tarjeta económica que hacía potentes cálculos paralelos y enviar los resultados a la memoria de la computadora podría tener aplicaciones más allá de los juegos alrededor de 2001 el acrónimo GPGPU (computación de propósito general en unidades de procesamiento de gráficos).

NVIDIA GPU Models.
NVIDIA produces three classes of GPU:
Los modelos de la marca GeForce GTX, GeForce RTX o Titan; estos son los menos costosos y están dirigidos al mercado del juego. Normalmente, estos modelos tienen menos compatibilidad con FP64 que las versiones científicas equivalentes y no utilizan memoria EEC.
Los modelos de la marca Tesla están dirigidos al mercado de informática científica de alta gama, tienen buen soporte FP64 y utilizan memoria EEC. Las tarjetas Tesla no tienen puertos de salida de video y no se pueden utilizar para jugar.
Las GPU de la marca Quadro; se trata esencialmente de GPU modelo Tesla con capacidad gráficas adicionales y están dirigidas al mercado de estaciones de trabajo de escritorio de alta gama.
Entre 2007 y la actualidad, NVIDIA ha introducido 8 generaciones de GPU diferentes y cada generación sucesiva tiene más funciones de software. Las generaciones llevan el nombre de científicos famosos y dentro de cada generación suele haber varios modelos que, a su vez, pueden diferir en las características del software.

Pascal Architecture.
Las GPU NVIDIA se construyen de forma jerárquica a partir de una gran cantidad de recursos informáticos básicos:
La unidad básica es un núcleo de cálculo simple; es capaz de realizar operaciones flotantes básicas de 32 bits operaciones puntuales y enteras.
Normalmente, los grupos de 32 núcleos se agrupan formando lo que NVIDIA llama “bloques de procesamiento de 32 núcleos”. Esto se debe a que, como se explica en las secciones de software, en un programa del kernel CUDA los hilos de ejecución están agrupados en grupos de 32 subprocesos que NVIDIA llama “warps” y que pueden considerarse como la unidad básica de ejecución en programas del kernel CUDA.
Los motores Warp se agrupan para formar lo que NVIDIA llama multiprocesadores simétricos o SM. Un SM suele tener 128 núcleos de cómputo. Los subprocesos en un programa del kernel CUDA se agrupan en una serie de subprocesos de tamaño fijo. El SM también agrega unidades de textura y varios recursos de memoria en el chip compartidos por igual.
Finalmente, se agrupan varios SM para formar la GPU final.
Para el mercado de los juegos, NVIDIA fabrica una gama de GPU que se diferencian por el número de unidades SM en el chip.

GPU Memory Types.
La memoria de la GPU también está organizada de forma jerárquica similar a los núcleos, con la GPU principal memoria en la base de la pirámide y varias memorias y caches especializados arriba. Los tipos de memoria son los siguientes:
Main memory: Es análoga a la memoria principal de una CPU; el programa en sí y todos los datos residen aquí. La CPU puede escribir y leer datos desde y hacia la memoria principal de la GPU. Estos intercambios se realizan a través del bus PCI y son relativamente lentos y, por lo tanto, los programas CUDA deben ser diseñados para minimizar las transferencias de datos. Se conservan útilmente los datos en la memoria principal de la GPU entre llamadas al kernel para que pueda ser reutilizado por llamadas sucesivas al kernel sin necesidad de recargar.
Constant Memory: 64 KB de memoria principal de GPU dedicada están reservados para datos constantes. La memoria constante tiene un caché dedicado que omite el caché L2, por lo que si todos los subprocesos desde un warp lee la misma ubicación de memoria, esto puede ser tan rápido como si los datos estuvieran en un registro.
Texture Memory: Esta característica está directamente relacionada con los orígenes del procesamiento de gráficos de GPU. La memoria de textura se utiliza para almacenar matrices de hasta tres dimensiones y está optimizada para el direccionamiento local de matrices 2D. Son de lectura y tienen sus propios caches dedicados.
Local memory: Estos son bloques de memoria privados para cada hilo de ejecución individual; se utilizan como almacenamiento de desbordamiento para variables locales en resultados temporales intermedios cuando los registros disponibles para un hilo son insuficientes.
Register file: Cada SM tiene registros de 64 K de 32 bits que son compartidos por igual por los bloques de subprocesos que se ejecutan simultáneamente en el SM. Esto puede considerarse como un punto muy importante de recurso de memoria. De hecho, existe un límite de 64 en el número máximo de warps simultáneos (equivalentes a 2K subprocesos) que se pueden ejecutar en un SM determinado.
Shared memory: Cada SM proporciona entre 32 KB y 64 KB de memoria compartida. Si un kernel requiere memoria compartida, el tamaño requerido se puede declarar en el momento del lanzamiento del kernel o en el momento de la compilación. Cada bloque de subprocesos que se ejecuta simultáneamente en un SM obtiene el mismo tamaño del bloque de memoria.
La memoria compartida es importante porque es muy rápida y porque proporciona la mejor manera de hilos dentro de un bloque de hilos para comunicarse entre sí.
Muchos de los primeros ejemplos de CUDA enfatizan el acceso a la memoria más rápido que proporciona la memoria compartida en comparación con el bajo rendimiento de la memoria principal (entonces mal almacenada en caché).
Las GPU actuales utilizan sus caches L1 y L2 junto con una alta ocupación para ocultar eficazmente la latencia de los accesos a la memoria principal. Los caches funcionan de manera más efectiva si los 32 subprocesos en un warp acceden a variables de 32 bits en hasta 32 ubicaciones de memoria adyacentes y la ubicación inicial está alineado en un límite de memoria de 32 palabras.

Warps and Waves.
La arquitectura de la GPU se refleja en la forma en que un kernel CUDA es diseñado y lanzado por software anfitrión. Diseñar buenos núcleos para resolver problemas particulares requiere habilidad y experiencia.
Si es nuevo en CUDA, puede esperar que establecer Nthreads igual a Ncores, la cantidad de núcleos en su GPU, sea suficiente para mantener la GPU completamente ocupada.
Para ser específicos, la GPU utilizada para la mayoría de nuestros ejemplos es una RTX 2070 que tiene 36 unidades SM (Nsm = 36) y cada SM tiene hardware para procesar dos warps de 32 hilos (Nwarp = 2).  Por lo tanto, para esta GPU Ncores = Nsm x Warp x 32 = 2304. Lo que es menos obvio es que durante el procesamiento del kernel cada unidad SM tiene una gran cantidad de subprocesos residentes, Nres, para la RTX 2070; Nres = 1024 equivalente a 32 warps.
Tenga en cuenta que, aunque la generación de GPU Turing tiene Nres = 1024, esto es inusual; todas las otras generaciones recientes de GPU NVIDIA tienen Nres = 2048, el doble de valor Turing. Dado que para estas GPU Warp = 2 es el mismo que para Turing, Nwaves será el doble del valor de Turing.

Blocks and Grids.
En CUDA el bloque de subprocesos es un concepto clave; es un grupo de subprocesos que se agrupan y se ejecutan en el mismo SM. El tamaño del bloque de hilo debe ser múltiplo del tamaño de warp (actualmente 32 para todas las GPU NVIDIA) hasta el tamaño máximo del hardware de 1024. En el kernel código, los subprocesos dentro del mismo bloque de subprocesos pueden comunicarse entre sí utilizando recursos compartidos o memoria global del dispositivo y pueden sincronizarse entre sí cuando sea necesario.
Cuando lanzamos un kernel CUDA especificamos la configuración de lanzamiento con dos valores, el tamaño del bloque de hilos y el número de bloques de hilos. La documentación CUDA se refiere a esto como lanzar una cuadrícula de bloques de subprocesos y el tamaño de la cuadrícula es solo el número de bloques de subprocesos.

Occupancy.
NVIDIA define la ocupación como la relación entre la cantidad de subprocesos que realmente residen en las unidades SM en comparación con el valor máximo Nres.
Incluso si lanzamos un kernel con suficientes subprocesos para lograr una ocupación del 100 por ciento, es posible que en realidad no logremos una ocupación total. La razón de esto es que cada SM tiene un número limitado de tamaño total de memoria compartida y un número limitado de registros. Si el tamaño de nuestro bloque de subprocesos es de 256,  entonces la ocupación total solo se logrará si cuatro (u ocho) bloques de subprocesos residen en cada SM, lo que reduce los recursos disponibles para cada bloque de subprocesos en el mismo factor.
Una ocupación inferior a la total no es necesariamente mala para el rendimiento, especialmente si el kernel está vinculado a la computación en lugar de a la memoria, es posible que deba aceptar una ocupación más baja si su kernel necesita cantidades significativas de memoria compartida.
El código kernel puede usar las variables integradas para determinar el rango de un hilo en su bloque de hilos y en la cuadrícula general.

Thinking and Coding in Parallel.
Las computadoras siempre han sido muy buenas para dar a los usuarios la impresión de que pueden realizar múltiples tareas al mismo tiempo.
Si tiene una PC de 4 núcleos más reciente, puede iniciar cuatro instancias de su largo cálculo con diferentes valores de parámetros para realizar realmente cálculos paralelos sin ningún esfuerzo de programación adicional.
Desafortunadamente, el enfoque de programación trivial no funciona en GPU que tienen núcleos de procesamiento muy simples diseñados para trabajar juntos en una sola tarea. La verdadera programación paralela requiere solo esto: muchos núcleos de procesamiento trabajando juntos para completar una sola tarea.

Flynn's Taxonomy.
Los científicos informáticos reconocen un pequeño número de arquitecturas informáticas en serie y paralelas descritas mediante acrónimos de cuatro letras resumidos en la taxonomía de Flynn.
El primero, el caso SISD, representa un procesador único “normal” que ejecuta un solo subproceso.
El segundo caso, SIMD, cubre arquitecturas donde el hardware puede ejecutar la misma instrucción en múltiples elementos de datos al mismo tiempo. Esto se puede lograr teniendo múltiples ALU alimentadas con diferentes elementos de datos pero utilizando un decodificador de instrucciones común.
El tercero, MIMD, es en realidad solo un conjunto de CPU independientes que realizan tareas independientes. Este caso incluye tanto las PC multinúcleo modernas que ejecutan Linux o Windows como los clusters de PC.
El cuarto, MISD, se incluye para mayor exhaustividad y rara vez se utiliza. Podría usarse para sistemas integrados especializados que requieran redundancia contra fallas.
El último, SIMT, fue presentado por NVIDIA como una variación de SIMD para describir su arquitectura GPU. Aunque ambos se utilizan para abordar cálculos científicos similares, existen diferencias entre ellos. En el modelo SIMD, un número relativamente pequeño de subprocesos utiliza hardware vectorial para procesar datos. En el modelo SIMT se utiliza una gran cantidad de subprocesos para procesar datos individuales.
En el caso SIMD/T el que resulta de interés para la programación paralela. Buscamos secciones de nuestro código donde se realizan la misma operación en múltiples elementos de datos: candidatos obvios son para bucles.
Hacer que sin_host sea global para todos los subprocesos, si bien es sencillo de implementar, introduce otra complicación más: si dos o más subprocesos intentan actualizar la variable simultáneamente ¡El resultado será indefinido! Con CUDA, un subproceso tendrá éxito y los intentos de otros subprocesos de actualización simultánea se ignorarán, por lo que la respuesta final será incorrecta.

Kernel Call Syntax.
La forma general de una llamada a un kernel CUDA utiliza hasta cuatro argumentos especiales en los <<< >>> corchetes y el propio núcleo pueden tener varios argumentos de función. Los cuatro argumentos dentro de los corchetes <<< >>> en orden son:
Primero: define las dimensiones de la cuadrícula de bloques de hilos utilizados por el núcleo.
Segundo: define el número de subprocesos en un solo bloque de subprocesos. 
Tercero: un argumento opcional de tipo size_t (o int) que define el número de bytes de memoria compartida asignada dinámicamente utilizados por cada bloque de subprocesos del kernel. No se reserva memoria compartida si este argumento se omite o se establece en cero.
Cuarto: un argumento opcional de tipo cudaStream_t que especifica la secuencia CUDA en cual ejecutar el kernel. Esta opción solo es necesaria en aplicaciones avanzadas que ejecutan múltiples núcleos simultáneos.

Latency Hiding and Occupancy.
Cuando un nuevo kernel comienza a ejecutarse, una deformación de 32 subprocesos comenzará a ejecutarse en cada uno de los motores warp en la GPU. Estas deformaciones se convierten en deformaciones activas y permanecen residentes hasta que todas las deformaciones en su bloque de hilos estén completas.
Una característica clave del diseño de la GPU es que cada motor warp puede procesar varios warps activos de forma intercalada; si uno de sus warps activos se detiene, un motor warp cambiará a otro warp activo capaz de funcionar sin pérdida de ciclos. La conmutación eficiente entre warps es posible porque cada hilo en un warp activo mantiene su propio estado y conjunto de registros.
Los factores que pueden limitar la ocupación son el tamaño del bloque de subprocesos, el número de bloques de subprocesos, el número de registros utilizados por cada subproceso y la cantidad de memoria compartida utilizada por un bloque de subprocesos.
La ocupación total es más importante para los kernels vinculados a la memoria que para los kernels vinculados a la computación, pero siempre es una buena idea mantener el código del kernel compacto y sencillo, ya que esto permitirá que el compilador asigne registros de manera más efectiva.

Shared Memory.
La memoria compartida es un grupo de memoria de acceso rápido de un tamaño típico de 64 KB disponible en cada SM.
Cada bloque de subprocesos que se ejecuta en un SM obtiene una asignación separada del tamaño requerido del grupo de memoria compartido.
Como su nombre lo indica, la memoria compartida es compartida por todos los subprocesos en un bloque de subprocesos, es decir, cualquier parte de ella puede ser leída o escrita por cualquier subproceso en el bloque de subprocesos.
La asignación de memoria compartida puede ser estática o dinámica.

Conclusión.
Leer este libro me ayudó a comprender de mejor manera la relación entre hardware y software que recae en las capacidades de hacer procesos de las computadoras y otros artefactos electrónicos, siendo el mismo hardware el que otorga las limitaciones de software.
