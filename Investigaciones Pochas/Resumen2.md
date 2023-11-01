Professional CUDA C Programming

CUDA Execution Model.
En general, un modelo de ejecución proporciona una visión operativa de cómo se ejecutan las instrucciones en una arquitectura informática específica. El modelo de ejecución CUDA expone una vista abstracta de la arquitectura paralela de GPU, lo que le permite razonar sobre la concurrencia de subprocesos.

GPU Architecture Overview.
La arquitectura de la GPU se basa en una matriz escalable de multiprocesadores de transmisión (SM). El paralelismo del hardware de la GPU se logra mediante la replicación de este bloque arquitectónico.
CUDA Cores.
Shared Memory / L1 Cache.
Register File.
Load / Store Units.
Warp Scheduler.
Cada SM en una GPU está diseñado para admitir la ejecución simultánea de cientos de subprocesos y, por lo general, hay varios SM por GPU, por lo que es posible tener miles de subprocesos ejecutándose simultáneamente en una sola GPU. Cuando se inicia una cuadrícula del núcleo, los bloques de subprocesos de esa cuadrícula del núcleo se distribuyen entre los SM disponibles para su ejecución. Una vez programados en un SM, los subprocesos de un bloque de subprocesos se ejecutan simultáneamente solo en ese SM asignado.
CUDA emplea una arquitectura de subprocesos múltiples de instrucción única (SIMT) para administrar y ejecutar subprocesos en grupos de 32 llamados warp. Todos los hilos de un warp ejecutan la misma instrucción al mismo tiempo.
La arquitectura SIMT es similar a la arquitectura SIMD (Single instruction, Multiple Data). Tanto SIMD como SIMT implementan el paralelismo transmitiendo la misma instrucción a múltiples unidades de ejecución. Una diferencia clave es que SIMD requiere que todos los elementos vectoriales en un vector se ejecuten juntos en un grupo sincrónico unificado, mientras que SIMT permite que múltiples subprocesos en el mismo warp que se ejecutan de forma independiente. El modelo SIMT incluye tres características clave que SIMD no incluye:
Cada hilo tiene su propio contador de direcciones de instrucciones.
Cada hilo tiene su propio estado de registro.
Cada hilo puede tener una ruta de ejecución independiente.
Un bloque de subprocesos está programado en un solo SM. Una vez que se programa un bloque de subprocesos en un SM, permanece allí hasta que se completa la ejecución.
La memoria y los registros compartidos son recursos valiosos en un SM. La memoria compartida se divide entre bloques de subprocesos residentes en el SM y los registros se dividen entre subprocesos. Los subprocesos de un bloque de subprocesos de un bloque de subprocesos pueden cooperar y comunicarse entre sí a través de estos recursos.
Compartir datos entre subprocesos paralelos puede provocar una condición de carrera: varios subprocesos acceden a los mismos datos con un orden indefinido, lo que da como resultado un comportamiento impredecible del programa.
Si bien las deformaciones dentro de un bloque de subprocesos se pueden programar en cualquier orden, la cantidad de deformaciones activas está limitada por los recursos de SM. Cuando un warp está inactivo por cualquier motivo (por ejemplo, esperando que se lean valores de la memoria del dispositivo), el SM es libre de programar otro warp disponible desde cualquier bloque de subprocesos que resida en el mismo SM.

A Magic Number: 32.
El número 32 es un número mágico en la programación CUDA. Proviene del hardware y tiene un impacto significativo en el rendimiento del software.
Conceptualmente, puedes considerarlo como la granularidad del trabajo procesado simultáneamente en forma SIMD por un SM. La optimización de sus cargas de trabajo para que se ajusten a los límites de un warp (grupo de 32 subprocesos) generalmente conducirá a una utilización más eficiente de los recursos informáticos de la GPU.

SM: The Heart Of The GPU Architecture.
El Streaming Multiprocessor (SM) es el corazón de la arquitectura GPU. Los registros y la memoria compartida son recursos escasos en el SM. CUDA divide estos recursos entre todos los subprocesos residentes en un SM. Por lo tanto, estos recursos limitados imponen una restricción estricta en el número de warps activos en un SM, que corresponde a la cantidad de paralelismo posible en un SM.

The Fermi Architecture.
La arquitectura Fermi fue la primera arquitectura informática GPU completa que ofrece las funciones necesarias para las aplicaciones HPC más exigentes.
Fermi cuenta con hasta 512 núcleos aceleradores, llamados núcleos CUDA. Cada núcleo CUDA tiene una unidad lógica aritmética entera (ALU) completamente canalizada y una unidad de punto flotante (FPU) que ejecuta una instrucción de punto flotante o entero por ciclo de reloj. Los núcleos CUDA están organizados en 16 multiprocesadores de transmisión (SM), cada uno con 32 núcleos CUDA.
Fermi incluye una caché L2 coherente de 768 KB, compartida por los 16 SM. 
Cada multiprocesador tiene 16 unidades de carga / almacenamiento, lo que permite calcular las direcciones de origen y destino para 16 subprocesos (un medio warp) por ciclo de reloj.
Cada SM cuenta con dos programadores warp y dos unidades de envío de instrucciones. Cuando se asigna un bloque de hilos a un SM, todos los hilos de un bloque de hilos se dividen en warps. Los dos programadores de warp seleccionan dos deformaciones y emiten una instrucción de cada deformación a un grupo de 16 núcleos CUDA, 16 unidades de carga / almacenamiento o 4 unidades de funciones especiales.
Una característica clave de Fermi es la memoria configurable en chip de 64 KB, que está dividida entre la memoria compartida y la caché L1. Para muchas aplicaciones de alto rendimiento, la memoria compartida es un factor clave para el rendimiento. La memoria compartida permite que los subprocesos dentro de un bloque cooperen, facilita la reutilización extensiva de datos en el chip y reduce en gran medida el tráfico fuera del chip.
Fermi también admite la ejecución simultánea del kernel: varios kernels iniciados desde el mismo contexto de aplicación ejecutándose en la misma GPU al mismo tiempo. La ejecución concurrente del kernel permite que los programas que ejecutan varios kernels pequeños utilicen completamente la GPU.

The Kepler Architecture.
La arquitectura GPU de Kepler, lanzada en el otoño de 2012, es una arquitectura informática rápida, altamente eficiente y de alto rendimiento. Tres innovaciones importantes en la arquitectura Kepler son:
SM mejorados.
Paralelismo dinámico.
Hyper-Q.
En el corazón del Kepler K20X se encuentra una nueva unidad SM, que comprende varias innovaciones arquitectónicas que mejoran la programabilidad y la eficiencia energética.
Cada Kepler SM incluye cuatro programadores warp y ocho despachadores de instrucciones, lo que permite emitir y ejecutar cuatro warps simultáneamente en un solo SM. La arquitectura Kepler K20X (capacidad de cálculo 3.5) puede programar 64 deformaciones por SM para un total de 2048 subprocesos residentes en un solo SM a la vez.
El paralelismo dinámico es una nueva característica introducida con las GPU Kepler que permite a la GPU lanzar dinámicamente nuevas cuadrículas (grids). Con esta característica, cualquier kernel puede iniciar otro kernel y administrar cualquier dependencia entre kernels necesaria para realizar correctamente el trabajo adicional. Esta característica facilita la creación y optimización de patrones de ejecución recursivos y dependientes de datos.
Hyper-Q agrega más conexiones de hardware simultáneas entre la CPU y la GPU, lo que permite que los núcleos de la CPU ejecute más tareas simultáneamente en la GPU. Como resultado, se puede esperar una mayor utilización de la GPU y un menor tiempo de inactividad de la CPU al utilizar las GPU Kepler.

Profile-Driven Optimization.
La elaboración de perfiles es el acto de analizar el desempeño del programa midiendo:
La complejidad espacial (memoria) o temporal del código de la aplicación.
El uso de instrucciones particulares.
La frecuencia y duración de las llamadas a funciones.
La creación de perfiles es un paso fundamental en el desarrollo de programas, especialmente para optimizar el código de la aplicación HPC. El desarrollo de una aplicación HPC normalmente implica dos pasos principales:
Desarrollar el código para que sea correcto.
Mejorar el código para mejorar el rendimiento.
El desarrollo basado en perfiles es particularmente importante en la programación CUDA por las siguientes razones: 
Una implementación ingenua del kernel generalmente no produce el mejor rendimiento. Las herramientas de creación de perfiles pueden ayudarle a encontrar las regiones críticas de su código que constituyen cuellos de botella en el rendimiento.
CUDA divide los recursos informáticos en un SM entre múltiples bloques de subprocesos residentes. Esta partición hace que algunos recursos se conviertan en limitadores de rendimiento. Las herramientas de creación de perfiles pueden ayudarle a obtener información sobre cómo se utilizan los recursos informáticos.
CUDA proporciona una abstracción de la arquitectura de hardware que le permite controlar la concurrencia de subprocesos. Las herramientas de creación de perfiles pueden ayudarle a medir, visualizar y guiar sus optimizaciones.
Las herramientas de creación de perfiles brindan información detallada sobre el rendimiento del kernel y lo ayudan a identificar cuellos de botella en los kernels.
nvvp es un perfilador visual que le ayuda a visualizar y optimizar el rendimiento de su programa CUDA. Esta herramienta muestra una línea de tiempo de la actividad del programa tanto en la CPU como en la GPU, lo que le ayuda a identificar oportunidades para mejorar el rendimiento.
nvprof recopila y muestra datos de perfiles en la línea de comando. nvprof se introdujo con CUDA 5 y evolucionó a partir de una herramienta de creación de perfiles CUDA de línea de comandos anterior.
Además de las métricas predefinidas, también puede definir sus propias métricas basadas en los contadores de hardware recopilados por el generador de perfiles.
Para identificar el cuello de botella en el rendimiento de un kernel, es importante elegir las métricas de rendimiento adecuadas y comparar el rendimiento medido con el rendimiento máximo teórico.
Hay tres limitadores comunes del rendimiento de un kernel que puede encontrar:
Ancho de banda de memoria.
Recursos informáticos.
Latencia de instrucción y memoria.

Events And Metrics.
En la creación de perfiles CUDA, un evento es una actividad contable que corresponde a un contador de hardware recopilado durante la ejecución del kernel. Una métrica es una característica de un kernel calculada a partir de uno o más eventos. Tenga en cuenta los siguientes conceptos sobre eventos y métricas:
La mayoría de los contadores se informan por multiprocesador de transmisión, pero no por toda la GPU.
Una sola carrera solo puede recolectar unos pocos contadores.
Es posible que los valores de los contadores no sean exactamente los mismos en ejecuciones repetidas debido a variaciones en la ejecución de la GPU (como el bloqueo de subprocesos y el orden de programación de deformación).

Knowing Hardware Resource Details.
Como programador de C, cuando escribe código solo para que sea correcto, puede ignorar con seguridad el tamaño de la línea de caché; sin embargo, al ajustar el código para obtener el máximo rendimiento, debe considerar las características de la caché en la estructura de su código.
Como programador de CUDA C, debe tener cierto conocimiento de los recursos de hardware si desea mejorar el rendimiento del kernel.
Si no comprende la arquitectura del hardware, el compilador CUDA seguirá haciendo un buen trabajo optimizando su kernel, pero no puede hacer mucho.

Understanding The Nature Of Warp Execution.
Al iniciar un kernel, ¿que ve desde el punto de vista del software? Para usted, parece que todos los subprocesos del kernel se ejecutan en paralelo. Desde un punto de vista lógico esto es cierto, pero desde el punto de vista del hardware no todos los subprocesos pueden ejecutarse físicamente en paralelo al mismo tiempo.

Warps And Thread Blocks.
Los warps son la unidad básica de ejecución en un SM. Cuando se inicia una cuadrícula de bloques de subprocesos, los bloques de subprocesos en la cuadrícula se distribuyen entre los SM. Una vez que un bloque de subprocesos está programado para un SM, los subprocesos en el bloque de subprocesos se dividen en warps.
Los bloques de hilos se pueden configurar para que sean unidimensionales, bidimensionales o tridimensionales. Sin embargo, desde la perspectiva del hardware, todos los hilos están dispuestos de forma unidimensional. Cada hilo tiene una identificación única en un bloque.
El diseño lógico de un bloque de hilo bidimensional o tridimensional se puede convertir en su diseño físico unidimensional utilizando la dimensión x como la dimensión más interna, la dimensión y como la segunda dimensión y la dimensión z como la más externa.
Por lo tanto, el hardware siempre asigna un número discreto de warps para un bloque de hilo. Un warp nunca se divide entre diferentes bloques de hilos. Si el tamaño del bloque de hilo no es un múltiplo par del tamaño del warp, algunos hilos del último warp quedan inactivos.

Thread Block: Logical View Versus Hardware View.
Desde una perspectiva lógica, un bloque de subprocesos es una colección de subprocesos organizados en un diseño 1D, 2D o 3D.
Desde la perspectiva del hardware, un bloque de hilos es una colección 1D de warps. Los hilos de un bloque de hilos se organizan en un diseño 1D y cada conjunto de 32 hilos consecutivos forma un warp. 

Warp Divergence.
El flujo de control es una de las construcciones fundamentales en cualquier lenguaje de programación de alto nivel.
Las CPU incluyen hardware complejo para realizar predicciones de bifurcación (branch), es decir, para predecir en cada verificación condicional que bifurcación tomará el flujo de control de una aplicación. Si la predicción es correcta, la bifurcación en CPU solo genera una pequeña penalización en el rendimiento. Si la predicción no es correcta, la CPU puede detenerse durante varios ciclos mientras se vacía la canalización de instrucciones.
Las GPU son dispositivos comparativamente simples sin complejos mecanismos de predicción de bifurcaciones. Debido a que todos los subprocesos de un warp deben ejecutar instrucciones idénticas en el mismo ciclo, si un subproceso ejecuta una instrucción, todos los subprocesos del warp deben ejecutar esa instrucción.
Si los subprocesos de un warp divergen, el warp ejecuta en serie cada ruta de bifurcación, deshabilitando los subprocesos que no tomen esa ruta. La divergencia de warps puede causar una degradación significativa del rendimiento.
Tenga en cuenta que la divergencia de las ramas ocurre solo dentro de un warp. Diferentes valores condicionales en diferentes deformaciones no causan divergencia de deformación.
Para obtener el mejor rendimiento, debes evitar diferentes rutas de ejecución dentro del mismo warp. Tenga en cuenta que la asignación de warps de los hilos en un bloque de hilos es determinista.
Si entrelaza datos utilizando un enfoque warp (en lugar de un enfoque thread), puede evitar la divergencia warp y lograr una utilización del dispositivo del 100 por ciento.

Key Reminders.
La divergencia de deformación ocurre cuando los subprocesos dentro de un warp toman diferentes rutas de código.
Se ejecutan en serie diferentes ramas if-then-else.
Intente ajustar la granularidad de la rama para que sea un múltiplo del tamaño del warp para evitar la divergencia del warp.
Diferentes warps pueden ejecutar código diferente sin penalización en el rendimiento.

Resource Partitioning.
El contexto de ejecución local de un warp consta principalmente de los siguientes recursos:
Contadores de programa.
Registros.
Memoria compartida.
El contexto de ejecución de cada warp procesado por un SM se mantiene en el chip durante toda la vida útil del warp.
Cada SM tiene un conjunto de registros de 32 bits almacenados en un archivo de registro que se dividen entre subprocesos y una cantidad fija de memoria compartida que se divide entre bloques de subprocesos.
La disponibilidad de recursos generalmente limita la cantidad de bloques de subprocesos residentes por SM. La cantidad de registros y la cantidad de memoria compartida por SM varían para dispositivos con diferente capacidad informática.
Un bloque de subprocesos se denomina bloque activo cuando se le han asignado recursos informáticos, como registros y memoria compartida. Los warps activos se pueden clasificar en los siguientes tres tipos:
Warp seleccionado.
Warp estancado.
Warp elegible.
Los programadores de warp en un SM seleccionan warps activos en cada ciclo y los envían a las unidades de ejecución. Un warp que se está ejecutando activamente se denomina warp seleccionado. Si un warp activo está listo para su ejecución pero no se está ejecutando actualmente es un warp elegible. Si un warp no está listo para su ejecución, es un warp estancado. Un warp es elegible para ejecución si se cumplen las dos condiciones siguientes: 
Treinta y dos núcleos CUDA están disponibles para su ejecución.
Todos los argumentos de la instrucción actual están listos.
La partición de recursos informáticos requiere una atención especial en la programación CUDA: los recursos informáticos limitan el número de warps activos.

Latency Hiding.
Un SM se basa en el paralelismo a nivel de subproceso para maximizar la utilización de sus unidades funcionales. Por lo tanto, la utilización está directamente relacionada con el número de warps residentes. El número de ciclos de reloj entre la emisión de una instrucción y su finalización se define como latencia de instrucción.
En comparación con la programación C en la CPU, la ocultación de la latencia es particularmente importante en programación CUDA. Los núcleos de CPU están diseñados para minimizar la latencia de uno o dos subprocesos a la vez, mientras que las GPU están diseñadas para manejar una gran cantidad de subprocesos ligeros y simultáneos para maximizar el rendimiento.
Al considerar la latencia de las instrucciones, las instrucciones se pueden clasificar en dos tipos básicos:
Instrucciones aritméticas.
Instrucciones de memoria.
La latencia de una instrucción aritmética es el tiempo entre el inicio de una operación aritmética y su salida. La latencia de instrucción de memoria es el tiempo entre que se emite una operación de carga o almacenamiento y que los datos llegan a su destino. Las latencias correspondientes para cada caso son aproximadamente:
10 - 20 ciclos para operaciones aritméticas.
400 - 800 ciclos para accesos a memoria global.
Para operaciones aritméticas, el paralelismo requerido se puede expresar como el número de operaciones necesarias para ocultar la latencia aritmética.
El rendimiento se especifica en número de operaciones por ciclo por SM, y un warp que ejecuta una instrucción corresponde a 32 operaciones. Por lo tanto, la cantidad requerida de warps por SM para mantener la utilización completa de los recursos informáticos se puede calcular para las GPU Fermi cómo 640 / 32 = 20 warps.
Para las operaciones de memoria, el paralelismo requerido se expresa como la cantidad de bytes por ciclo necesarios para ocultar la latencia de la memoria.
Debido a que el rendimiento de la memoria generalmente se expresa en gigabytes por segundo, primero debe convertir el rendimiento en gigabytes por ciclo utilizando la frecuencia de memoria correspondiente.
Un ejemplo de frecuencia de memoria Fermi (medida en un Tesla C2070) es 1,566 GHz. Un ejemplo de frecuencia de memoria Kepler (medida en un Tesla K20) es 1,6 GHz.
Al multiplicar los bytes por ciclo por la latencia de la memoria, se obtiene el paralelismo necesario para las operaciones de memoria Fermi en casi 74 KB de I/O de memoria en vuelo para lograr una utilización completa.
La conexión de estos valores con los warps o el número de hilos depende de la aplicación.
La arquitectura Fermi tiene 16 SM.
Al igual que la latencia de las instrucciones, puede aumentar el paralelismo disponible creando más operaciones de memoria independientes dentro de cada subproceso / warp, o creando más subprocesos / warps activos simultáneamente.
La ocultación de la latencia depende del número de warps activos por SM, que está implícitamente determinado por la configuración de ejecución y las restricciones de recursos (registros y uso de memoria compartida en un kernel).

Throughput And Bandwidth.
El ancho de banda y el rendimiento a menudo se confunden, pero pueden usarse indistintamente según la situación.
El ancho de banda se utiliza normalmente para hacer referencia a un valor máximo teórico, mientras que el rendimiento se utiliza para hacer referencia a un valor alcanzado.
El ancho de banda se usa generalmente para describir la mayor cantidad posible de transferencia de datos por unidad de tiempo, mientras que el rendimiento se puede usar para describir la tasa de cualquier tipo de información u operaciones realizadas por unidad de tiempo, como por ejemplo, cuantas instrucciones se completan por ciclo.

Exposing Sufficient Parallelism.
Debido a que las particiones de GPU calculan recursos entre subprocesos, el cambio entre warps concurrentes tiene muy poca sobrecarga (del orden de uno o dos ciclos), ya que el estado requerido ya está disponible en el chip. Si hay suficientes subprocesos activos simultáneamente, puede mantener la GPU ocupada en cada etapa del proceso en cada ciclo.
Una fórmula simple para calcular el paralelismo requerido es multiplicar el multiplicar el número de núcleos por SM por la latencia de una instrucción aritmética en ese SM.

Occupancy.
Las instrucciones se ejecutan secuencialmente dentro de cada núcleo CUDA. Cuando un warp se detiene, el SM pasa a ejecutar otros warps elegibles. 
Después de haber especificado la capacidad de cálculo, los datos en la sección de límites físicos se completan automáticamente. A continuación, debe ingresar la siguiente información de recursos del kernel:
Hilos por bloque (configuración de ejecución).
Registros por hilo (uso de recursos).
Memoria compartida por bloque (uso de recursos).
Una vez que se han ingresado estos datos, la ocupación de su kernel se muestra en la sección de Datos de ocupación de GPU.
La cantidad de registros utilizados por un núcleo puede tener un impacto significativo en la cantidad de warps residentes.
Para mejorar su ocupación, es posible que también necesite cambiar el tamaño de la configuración del bloque de subprocesos o reajustar el uso de recursos para permitir más deformaciones activas simultáneamente y mejorar la utilización de los recursos informáticos.
Aunque cada caso alcanzará diferentes límites de hardware, ambos provocan que los recursos informáticos se subutilicen y dificultan la creación de un paralelismo suficiente para ocultar la latencia de las instrucciones y la memoria.

Guidelines For Grid And Block Size.
El uso de estas pautas ayudará a que su aplicación escale en dispositivos actuales y futuros:
Mantenga el número de hilos por bloque en un múltiplo del tamaño del warp (32).
Evite tamaños de bloques pequeños: comience con al menos 128 o 258 subprocesos por bloque.
Ajuste el tamaño del bloque hacia arriba o hacia abajo según los requisitos de recursos del kernel.
Mantenga la cantidad de bloques mucho mayor que la cantidad de SM para exponer suficiente paralelismo en su dispositivo.
Realice experimentos para descubrir la mejor configuración de ejecución y uso de recursos.

Synchronization.
La sincronización de barrera es una primitiva común en muchos lenguajes de programación paralelos. En CUDA, la sincronización se puede realizar en dos niveles:
A nivel del sistema: Espera a que se complete todo el trabajo tanto en el host como en el dispositivo.
Nivel de bloque: Espera a que todos los subprocesos de un bloque de subprocesos alcancen el mismo punto de ejecución en el dispositivo.
Dado que muchas llamadas a la API de CUDA y todos los lanzamientos del kernel son asíncronos con respecto al host, se puede usar cudaDeviceSynchronize para bloquear la aplicación del host hasta que se hayan completado todas las operaciones de CUDA (copias, kernels, etc.): cudaError_t cudaDeviceSynchronize(void);
Esta función puede devolver errores de operaciones CUDA asíncronas anteriores.
Debido a que las deformaciones en un bloque de subprocesos se ejecutan en un orden indefinido, CUDA proporciona la capacidad de sincronizar su ejecución con una barrera local del bloque.
Cuando se llama a _syncthreads, cada subproceso en el mismo bloque de subprocesos debe esperar hasta que todos los demás subprocesos en ese bloque de subprocesos hayan alcanzado este punto de sincronización.
Los subprocesos dentro de un bloque de subprocesos pueden compartir datos a través de registros y memoria compartidos. Al compartir datos entre subprocesos, debe tener cuidado para evitar condiciones de carrera. Las condiciones de carrera, o peligros, son accesos desordenados de múltiples subprocesos a la misma ubicación de memoria.
No hay sincronización de subprocesos entre diferentes bloques. La única forma segura de sincronizar entre bloques es utilizar el punto de sincronización global al final de cada ejecución del kernel; es decir, finalizar el kernel actual e iniciar un nuevo kernel para el trabajo que se realizará después de la sincronización global.
Al no permitir que los subprocesos de diferentes bloques se sincronicen entre sí, las GPU pueden ejecutar bloques en cualquier orden.

Scalability.
La escalabilidad es una característica deseable para cualquier aplicación paralela. La escalabilidad implica que proporcionar recursos de hardware adicionales a una aplicación paralela produce una aceleración en relación con la cantidad de recursos agregados. Un programa paralelo escalable utiliza todos los recursos informáticos de manera eficiente para mejorar el rendimiento.
La capacidad de ejecutar el mismo código de aplicación en una cantidad variable de núcleos informáticos se denomina escalabilidad transparente. Una plataforma transparente escalable amplía los casos de uso de las aplicaciones existentes y reduce la carga para los desarrolladores porque pueden evitar realizar cambios para hardware nuevo o diferente.
Cuando se inicia un kernel CUDA, los bloques de subprocesos se distribuyen entre varios SM.

Avoiding Branch Divergence.
A veces, el flujo de control depende de los índices de hilo. La ejecución condicional dentro de un warp puede provocar una divergencia del warp que puede provocar un rendimiento deficiente del kernel. Al reorganizar los patrones de acceso a los datos, puede reducir o evitar la divergencia de deformación.

The Parallel Reduction Problem.
Una forma común de lograr la suma paralela es utilizar una implementación iterativa por pares: un fragmento contiene solo un par de elementos y un hilo suma esos dos elementos para producir un resultado parcial.
Dependiendo de donde se almacenan los elementos de salida en el lugar para cada iteración, las implementaciones de suma paralela por pares se pueden clasificar en los dos tipos siguientes:
Par de vecinos: Los elementos se emparejan con su vecino inmediato.
Par entrelazado: Los elementos emparejados están separados por una zancada determinada.
Este problema general de realizar una operación conmutativa y asociativa a través de un vector se conoce como problema de reducción.

Divergence In Parallel Reduction.
La distancia entre dos elementos vecinos, zancada, se inicializa en 1 al principio. Después de cada ronda de reducción, esta distancia se multiplica por 2. Después de la primera ronda, los elementos pares de idata serán reemplazados por sumas parciales. Después de la segunda ronda, cada cuarto elemento de datos se sustituirá por sumas parciales adicionales.

Improving Divergence In Parallel Reduction.
Examine el kernel reduceNeighbored y observe la siguiente declaración condicional: if ((tid % (2 * stride)) == 0)
Debido a que esta afirmación sólo es cierta para hilos pares, provoca deformaciones muy divergentes. En la primera iteración de la reducción paralela, solo los subprocesos pares ejecutan el cuerpo de esta declaración condicional, pero todos los subprocesos deben programarse. En la segunda iteración, sólo una cuarta parte de todos los subprocesos están activos, pero aún así todos los subprocesos deben programarse. La divergencia de warps se puede reducir reorganizando el índice de matriz de cada subproceso para forzar a los subprocesos vecinos a realizar la suma.
Con un tamaño de bloque de hilo de 512 hilos, los primeros 8 warps ejecutan la primera ronda de reducción y los 8 warps restantes no hacen nada. En la segunda ronda, los primeros 4 warps ejecutan la reducción y los 12 warps restantes no hacen nada. Por lo tanto, no hay ninguna divergencia.

Reducing With Interleaved Pairs.
El enfoque de pares entrelazados invierte el paso de elementos en comparación con el enfoque de vecinos: el paso se inicia en la mitad del tamaño del bloque de hilos y luego se reduce a la mitad en cada iteración.
La implementación entrelazada es 1,69 veces más rápida que la primera implementación y 1,34 veces más rápida que la segunda implementación. Esta mejora de rendimiento es principalmente el resultado de la carga de memoria global y los patrones de almacenamiento en reduceInterleaved.

Unrolling Loops.
El desenrollado de bubbles es una técnica que intenta optimizar la ejecución del bucle reduciendo la frecuencia de las bifurcaciones y las instrucciones de mantenimiento del bucle. Al desenrollar un bucle, en lugar de escribir el cuerpo de un bucle una vez y utilizar un bucle para ejecutarlo repetidamente, el cuerpo se escribe en código varias veces. Cualquier bucle circundante reduce sus iteraciones o se elimina por completo.
La razón de las ganancias de rendimiento al desenrollar el bucle puede no ser evidente al observar el código de alto nivel. La mejora proviene de mejoras y optimizaciones de instrucciones de bajo nivel que el compilador realiza en el bucle desenrollado.
Desenrollar en CUDA puede significar una variedad de cosas. Sin embargo, el objetivo sigue siendo el mismo: mejorar el rendimiento reduciendo los gastos generales de instrucción y creando instrucciones más independientes para programar.

Reducing With Unrolling.
Puede notar que cada bloque de subprocesos en el kernel reduceInterleaved maneja solo una porción de los datos, que puede considerarse un bloque de datos.
Tenga en cuenta la siguiente declaracion agregada al inicio del kernel. Aqui, cada hilo agrega un elemento del bloque de datos vecino. Conceptualmente puedes pensar en esto como una iteracion del ciclo de reduccion que reduce entre bloques de datos. if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

Reducing With Unrolled Warps.
_syncthreads se utiliza para la sincronización dentro del bloque. En los núcleos de reducción, se utiliza en cada ronda para garantizar que todos los subprocesos que escriben resultados parciales en la memoria global se hayan completado antes de que cualquier subproceso continúe con la siguiente ronda.
Sin embargo, considere el caso en el que quedan 32 o menos hilos (es decir, un solo warp). Debido a que la ejecución de warp es SIMT, existe una sincronización intra-warp implícita después de cada instrucción.
Este desenrollado del warp evita la ejecución de la lógica de control de bucle y sincronización de subprocesos.
Tenga en cuenta que la variable vmem se declara con el calificador volátil, que le dice al compilador que debe almacenar vmem[tid] en la memoria global con cada asignación. Si se omite el calificador volátil, este código no funcionará correctamente porque el compilador o el caché pueden optimizar algunas lecturas o escrituras en la memoria global o compartida.

Reducing With Template Functions.
Si bien es posible desenrollar bucles manualmente, el uso de funciones de plantilla puede ayudar a reducir aún más la sobrecarga de las ramas. CUDA admite parámetros de plantilla en las funciones del dispositivo.
La única diferencia en comparación con reduceCompleteUnrollWarps8 es que reemplazo el tamaño del bloque con un parámetro de plantilla. Las declaraciones if que verifican el tamaño del bloque se evaluaran en tiempo de compilacion y se eliminaran si la condición no es verdadera, lo que resultará en un bucle interno muy eficiente.

Dynamic Parallelism.
El paralelismo dinámico permite un enfoque más jerárquico donde la concurrencia se puede expresar en múltiples niveles en un kernel de GPU. El paralelismo dinámico permite un enfoque más jerárquico donde la concurrencia se puede expresar en múltiples niveles en un kernel de la GPU.
Con el paralelismo dinámico, puede posponer la decisión de exactamente cuántos bloques y grids crear en una GPU hasta el tiempo de ejecución, aprovechando dinámicamente los programadores de hardware y los balanceadores de carga de la GPU y adaptándose en respuesta a decisiones o cargas de trabajo basadas en datos.
La capacidad de crear trabajo directamente desde la GPU también puede reducir la necesidad de transferir control de ejecución y datos entre el host y el dispositivo, ya que las decisiones de configuración de lanzamiento se pueden tomar en tiempo de ejecución mediante subprocesos que se ejecutan en el dispositivo.

Nested Execution.
Con el paralelismo dinámico, los conceptos de ejecución del kernel (grids, bloques, configuración de inicio, etc.) con los que ya está familiarizado también se pueden aplicar a la invocación del kernel directamente en la GPU.
En el paralelismo dinámico, las ejecuciones del kernel se clasifican en dos tipos: padre e hijo. Un hilo principal, un bloque de hilos principal o un grid principal han lanzado un nuevo grid, la cuadrícula hijo.
Los inicios de grid en un hilo de dispositivo son visibles a través de un bloque de hilos. Esto significa que un subproceso puede sincronizarse en los grids secundarios lanzados por ese subproceso o por otros subprocesos en el mismo bloque de subprocesos.
Cuando un padre inicia un grid secundario, no se garantiza que el hijo comience la ejecución hasta que el bloque de subprocesos principal se sincronice explícitamente con el hijo.
Los grids principales y secundarios comparten el mismo almacenamiento de memoria global y constante, pero tienen memoria local y compartida distinta. Los grids padre e hijo tienen acceso simultáneo a la memoria global, con garantías débiles de coherencia entre hijo y padre.
La memoria compartida y local son privadas para un bloque de subprocesos o un subproceso, respectivamente, y no son visibles ni coherentes entre padre e hijo.

Introducing The CUDA Memory Model.
El acceso y la gestión de la memoria son partes importantes de cualquier lenguaje de programación.
Debido a que muchas cargas de trabajo están limitadas por la rapidez con la que pueden cargar y almacenar datos, tener una gran cantidad de memoria de baja latencia y gran ancho de banda puede ser muy beneficioso para el rendimiento. Sin embargo, adquirir memoria de gran capacidad y alto rendimiento no siempre es posible ni económico.

Benefits Of A Memory Hierarchy.
En general, las aplicaciones no acceden a datos arbitrarios ni ejecutan código arbitrario en ningún momento. Hay dos tipos diferentes de localidad:
Localidad temporal (localidad en el tiempo).
Localidad espacial (localidad en el espacio).
La localidad temporal supone que si se hace referencia a una ubicación de datos, es más probable que se haga referencia a ella nuevamente dentro de un periodo de tiempo corto y es menos probable que se haga referencia a ella a medida que pasa más y más tiempo.
Las computadoras modernas utilizan una jerarquía de memoria de memorias de latencia progresivamente menor pero de menor capacidad para optimizar el rendimiento. Esta jerarquía de memoria solo es útil debido al principio de localidad. Una jerarquía de memoria consta de múltiples niveles de memoria con diferentes latencias, anchos de banda y capacidades. 
La memoria principal tanto para las CPU como para las GPU se implementa mediante DRAM (memoria dinámica de acceso aleatorio), mientras que la memoria de menor latencia (como la caché L1 de la CPU) se implementa mediante la SRAM (memoria estática de acceso aleatorio). El nivel más grande y más lento en la jerarquía de memoria generalmente se implementa mediante un disco magnético o una unidad flash.
Tanto las GPU como las CPU utilizan principios y modelos similares en el diseño de la jerarquía de memoria.

CUDA Memory Model.
Para los programadores, generalmente existen dos clasificaciones de memoria:
Programable: Usted controla explícitamente qué datos se colocan en la memoria programable.
No programable: No tiene control sobre la ubicación de los datos y depende de técnicas automáticas para lograr un buen rendimiento.
En la jerarquía de memoria de la CPU, la caché L1 y la caché L2 son ejemplos de memoria no programable. Por otro lado, el modelo de memoria CUDA te expone muchos tipos de memoria programable: 
Registros.
Memoria compartida.
Memoria local.
Memoria constante.
Memoria de textura.
Memoria global.

Registros.
Los registros son el espacio de memoria más rápido en una GPU. Una variable automática declarada en un kernel sin ningún otro calificador de tipo generalmente se almacena en un registro.
Las variables de registro son privadas para cada hilo. Un kernel normalmente utiliza registros para contener variables privadas de subprocesos a las que se accede con frecuencia.
Los registros son recursos escasos que se dividen entre warps activos en un SM. En las GPU Fermi, existe un límite de hardware de 63 registros por subproceso. Kepler amplia este límite a 255 registros por hilo.
Puede verificar los recursos de hardware utilizados por un kernel con la opción del compilador nvcc a continuación.
Si un kernel utiliza más registros que el límite de hardware, los registros sobrantes se extienden a la memoria local. Este derrame de registros puede tener consecuencias adversas en el rendimiento.
También puede controlar el número máximo de registros utilizados por todo los núcleos en una unidad de compilación utilizando la opción del compilador maxregcount.

Local Memory.
Las variables en un kernel que son elegibles para registros pero que no pueden caber en el espacio de registro asignado para ese kernel se derramaran en la memoria local. Las variables que probablemente el compilador coloque en la memoria local son: 
Matrices locales referenciadas con índices cuyos valores no se pueden determinar en tiempo de compilación.
Grandes estructuras o matrices locales que consumirían demasiado espacio de registro.
Cualquier variable que no se ajuste dentro del límite de registros del kernel.
El nombre “memoria local” es engañoso: los valores transferidos a la memoria local residen en la misma ubicación física que la memoria global, por lo que los accesos a la memoria local se caracterizan por una alta latencia y un bajo ancho de banda y están sujetos a los requisitos para un acceso eficiente a la memoria que se describen en la sección “Patrones de acceso a la memoria” que se encuentra más adelante en este capítulo.

Shared Memory.
Las variables decoradas con el siguiente atributo en un kernel se almacenan en la memoria compartida: _shared_.
Debido a que la memoria compartida está en el chip, tiene un ancho de banda mucho mayor y una latencia mucho menor que la memoria local o global.
Cada SM tiene una cantidad limitada de memoria compartida que se divide entre bloques de subprocesos.
La memoria compartida se declara en el alcance de una función del kernel pero comparte su vida útil con un bloque de subprocesos.
La memoria compartida sirve como un medio básico para la comunicación entre subprocesos. Los subprocesos dentro de un bloque pueden cooperar compartiendo datos almacenados en la memoria compartida.
Esta función crea una barrera que todos los subprocesos del mismo bloque de subprocesos deben alcanzar antes de permitir que cualquier otro subproceso continúe. Al crear una barrera para todos los subprocesos dentro de un bloque de subprocesos, que puede evitar un posible peligro para los datos.

Constant Memory.
La memoria constante reside en la memoria del dispositivo y se almacena en caché en una caché constante dedicada por SM.
Las variables constantes deben declararse con alcance global, fuera de cualquier núcleo. Se puede declarar una cantidad limitada de memoria constante: 64 KB para todas las capacidades informáticas. 
Los kernels solo pueden leer desde la memoria constante.
La memoria constante funciona mejor cuando todos los subprocesos de un warp leen desde la misma dirección de memoria. Por ejemplo, un coeficiente para una fórmula matemática es un buen caso de uso para la memoria constante porque todos los hilos en un warp usarán el mismo coeficiente para realizar el mismo cálculo con datos diferentes.

Texture Memory.
La memoria de textura reside en la memoria del dispositivo y se almacena en caché en una caché de sólo lectura por SM. La memoria de textura es un tipo de memoria global a la que se accede a través de una caché dedicada de solo lectura. La caché de sólo lectura incluye soporte para filtrado de hardware, que se puede realizar interpolación de punto flotante como parte del proceso de lectura.

Global Memory.
La memoria global es la memoria más grande, de mayor latencia y más utilizada en una GPU.
Una variable en la memoria global se puede declarar de forma estática o dinámica.
El host asigna la memoria global mediante cudaMalloc y el host la libera mediante cudaFree. Luego, los punteros a la memoria global se pasan a las funciones del kernel como parámetros. Las asignaciones de memoria global existen durante la vida útil de una aplicación y son accesibles para todos los subprocesos de todos los núcleos.
La memoria global reside en la memoria del dispositivo y se puede acceder a ella mediante transacciones de memoria de 32 bytes, 64 bytes o 128 bytes. Estas transacciones de memoria deben estar alineadas de forma natural; es decir, la primera dirección debe ser un múltiplo de 32 bytes, 64 bytes o 128 bytes.
En general, cuantas más transacciones sean necesarias para satisfacer una solicitud de memoria, mayor será la posibilidad de que se transfieran bytes no utilizados, lo que provocará una reducción en la eficiencia del rendimiento.
Para una solicitud de memoria warp determinada, el número de transacciones y la eficiencia del rendimiento están determinados por la capacidad de cálculo del dispositivo. Para dispositivos con capacidad informática 1.0 y 1.1, los requisitos de acceso a la memoria global son muy estrictos.

CPU Caches.
Al igual que las caches de CPU, las caches de GPU son memorias no programables. Hay cuatro tipos de cache en dispositivos GPU: 
L1.
L2.
Constante de solo lectura.
Textura de solo lectura.
Hay una caché L1 por SM y una caché L2 compartida por todos los SM. Tanto la caché L1 como la L2 se utilizan para almacenar datos en la memoria local y global, incluidos los derrames de registros.
En la CPU, tanto las cargas como los almacenes de memoria se pueden almacenar en caché.
Cada SM también tiene un caché constante de solo lectura y un caché de textura de solo lectura que se utilizan para mejorar el rendimiento de lectura desde sus respectivos espacios de memoria en la memoria del dispositivo.

Variables In File Scope: Visible Versus Accessible.
En la programación CUDA, trabaja en dos mundos distintos: el host y el dispositivo.
La API de tiempo de ejecución de CUDA puede acceder a variables tanto del host como del dispositivo, pero depende de usted proporcionar los argumentos correctos para las funciones correctas para que funcionen correctamente en las variables correctas.

Memory Management.
La gestión de la memoria en la programación CUDA es similar a la programación en C, con la responsabilidad adicional del programador de gestionar explícitamente el movimiento de datos entre el host y el dispositivo. Si bien NVIDIA se acerca metódicamente a unificar el espacio de memoria del host y del dispositivo con cada versión de CUDA, para la mayoría de las aplicaciones el movimiento manual de datos sigue siendo un requisito.
Para lograr el máximo rendimiento, CUDA proporciona funciones que preparan la memoria del dispositivo en el host y transfieren datos explícitamente hacia y desde el dispositivo.

Memory Allocation And Deallocation.
El modelo de programación CUDA supone un sistema heterogéneo que consta de un host y un dispositivo, cada uno con su propio espacio de memoria independiente. Las funciones del kernel operan en el espacio de memoria del dispositivo y el tiempo de ejecución de CUDA proporciona funciones para asignar y desasignar memoria del dispositivo. Puede asignar memoria global en el host utilizando la siguiente función: cudaError_t cudaMalloc(void **devPtr, size_t count);
Esta función asigna bytes de memoria global en el dispositivo y devuelve la ubicación de esa memoria en el puntero devPtr. La memoria asignada está adecuadamente alineada para cualquier tipo de variable, incluidos números enteros, valores de punto flotante, booleanos, etc. Es su responsabilidad llenar la memoria global asignada con datos transferidos desde el host o inicializarla con la siguiente función: cudaError_t cudaMemset(void *devPtr, int value, size_t count);
Esta función llena cada uno de los bytes de conteo comenzando en la dirección de memoria del dispositivo devPtr con el valor almacenado en la variable value.
Una vez que una aplicación ya no utiliza una parte de la memoria global asignada, se puede desasignar usando: cudaError_t cudaFree(void *devPtr);
La asignación y desasignación de memoria del dispositivo son operaciones costosas, por lo que las aplicaciones deben reutilizar la memoria del dispositivo siempre que sea posible para minimizar el impacto en el rendimiento general.

Pinned Memory.
La memoria del host asignada es paginable de forma predeterminada, es decir, está sujeta a operaciones de error de página que mueven datos en la memoria virtual del host a diferentes ubicaciones físicas según lo indique el sistema operativo.
La GPU no puede acceder de forma segura a los datos en la memoria del host paginable porque no tiene control sobre cuando el sistema operativo del host puede decidir mover físicamente esos datos.
El tiempo de ejecución de CUDA le permite asignar directamente memoria de host anclada usando: cudaError_t cudaMallocHost(void **devPtr, size_t count);
Esta función asigna bytes de memoria del host que están bloqueados en la página y son accesibles para el dispositivo. Dado que el dispositivo puede acceder directamente a la memoria fijada, se puede leer y escribir con un ancho de banda mucho mayor que la memoria paginable.

Memory Transfer Between The Host And Device.
La memoria fijada es más costosa de asignar y desasignar que la memoria paginable, pero proporciona un mayor rendimiento de transferencia para grandes transferencias de datos.
La aceleración lograda cuando se utiliza memoria fijada en relación con la memoria paginable depende de la capacidad informática del dispositivo.
La agrupación de muchas transferencias pequeñas en una transferencia más grande mejora el rendimiento porque reduce la sobrecarga por transferencia.
Las transferencias de datos entre el host y el dispositivo a veces pueden superponerse con la ejecución del kernel.

Zero-Copy Memory.
En general, el host no puede acceder directamente a las variables del dispositivo y el dispositivo no puede acceder directamente a las variables del host.
Los subprocesos de la GPU pueden acceder directamente a la memoria de copia cero. Existen varias ventajas al utilizar memoria de copia cero en los kernels CUDA, como por ejemplo: 
Aprovechar la memoria del host cuando no hay suficiente memoria del dispositivo.
Evitar la transferencia de datos explícita entre el host y el dispositivo.
Mejora de las tasas de transferencia PCle.
Cuando utilice memoria de copia cero para compartir datos entre el host y el dispositivo, debe sincronizar los accesos a la memoria entre el host y el dispositivo.
La memoria de copia cero es una memoria fija (no paginable) que se asigna al espacio de direcciones del dispositivo.
cudaHostAllocDefault hace que el comportamiento de cudaHostAlloc sea idéntico al de cudaMallocHost. La configuración de cudaHostAllocPortable devuelve memoria fija que puede ser utilizada por todos los contextos CUDA, no solo por el que realizó la asignación. El indicador cudaHostAllocWriteCombined devuelve memoria combinada de escritura, que se puede transferir a través del bus PCI Express más rápidamente en algunas configuraciones del sistema, pero la mayoría de los host no pueden leerla de manera eficiente.
Puede obtener el puntero del dispositivo para la memoria anclada asignada utilizando la siguiente función: cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
Esta función devuelve un puntero de dispositivo en pDevice al que se puede hacer referencia en el dispositivo para acceder a la memoria del host fijada y asignada.
El uso de memoria de copia cero como complemento de la memoria del dispositivo con operaciones frecuentes de lectura / escritura reducirá significativamente el rendimiento.
Hay dos categorías comunes de arquitecturas de sistemas informáticos heterogéneos: integradas y discretas.En las arquitecturas integradas, las CPU y GPU se fusionan en un solo chip y comparten físicamente la memoria principal.
Para sistemas discretos con dispositivos conectados al host a través del bus PCle, la memoria de copia cero sólo es ventajosa en casos especiales.
Debido a que la memoria anclada asignada se comparte entre el host y el dispositivo, debe sincronizar los accesos a la memoria para evitar posibles riesgos de datos causados por múltiples subprocesos que acceden a la misma ubicación de memoria sin sincronización.

Unified Virtual Addressing.
Los dispositivos con capacidad informática 2.0 y posteriores admiten un modo de direccionamiento especial llamado Direccionamiento virtual unificado (UVA).
Antes de UVA, era necesario administrar que punteros se referían a la memoria del host y cuales a la memoria del dispositivo.
Bajo UVA, la memoria de host fijada asignada con cudaHostAlloc tiene punteros de dispositivo y host idénticos. Por lo tanto, puede pasar el puntero devuelto directamente a una función del núcleo.

Unified Memory.
Con CUDA 6.0, se introdujo una nueva característica llamada Memoria Unificada para simplificar la administración de la memoria en el modelo de programación CUDA. La memoria unificada crea un grupo de memoria administrada, donde se puede acceder a cada asignación de este grupo de memoria tanto en la CPU como en la GPU con la misma dirección de memoria (es decir, puntero).
La memoria unificada depende de la compatibilidad con el direccionamiento virtual unificado (UVA), pero son tecnologías completamente diferentes. UVA proporciona un único espacio de direcciones de memoria virtual para todos los procesadores del sistema.
La memoria unificada ofrece un modelo de “puntero único a datos” que es conceptualmente similar a la memoria de copia cero. Sin embargo, la memoria de copia cero se asigna en la memoria del host y, como resultado, el rendimiento del kernel generalmente se ve afectado por los accesos de alta latencia a la memoria de copia cero a través del bus PCle.
La memoria administrada se refiere a asignaciones de memoria unificada que el sistema subyacente administra automáticamente y es interoperabilidad con asignaciones específicas del dispositivo, como las creadas mediante la rutina cudaMalloc.
La memoria administrada se puede asignar de forma estática o dinámica. Puede declarar estáticamente una variable de dispositivo como una variable administrada agregando una anotación _ managed _ a su declaración.

Memory Access Patterns.
La mayor parte del acceso a los datos del dispositivo comienza en la memoria global y la mayoría de las aplicaciones de GPU tienden a estar limitadas por el ancho de banda de la memoria.
Para lograr el mejor rendimiento al leer y escribir datos, las operaciones de acceso a la memoria deben cumplir ciertas condiciones. Una de las características distintivas del modelo de ejecución CUDA es que las instrucciones se emiten y ejecutan por warp. Las operaciones de memoria también se emiten por warp. Al ejecutar una instrucción de memoria, cada hilo en un warp proporciona una dirección de memoria que está cargando o almacenando.

Aligned And Coalesced Access.
Las cargas / almacenamiento de memoria global se organizan a través de caches. La memoria global es un espacio de memoria lógica al que se puede acceder desde su kernel.
Todos los accesos a la memoria global pasan por la caché L2. Muchos accesos también pasan por la caché L1, según el tipo de acceso y la arquitectura de su GPU. Si se utilizan caches L1 y L2, el acceso a la memoria se realiza mediante una transacción de memoria de 128 bytes.
Una línea de caché L1 tiene 128 bytes y se asigna a un segmento alineado de 128 bytes en la memoria del dispositivo.
Hay dos características de los accesos a la memoria del dispositivo por las que debe esforzarse al optimizar su aplicación:
Accesos a memoria alineados.
Accesos a memoria fusionados.
Los accesos a memoria alineados ocurren cuando la primera dirección de una transacción de memoria del dispositivo es un múltiplo par de la granularidad de la caché que se utiliza para dar servicio a la transacción (ya sea 32 bytes para la caché L2 o 128 bytes para la caché L1).
Los accesos a la memoria fusionados ocurren cuando los 32 subprocesos en una deformación acceden a una porción contigua de memoria.
Los accesos a memoria fusionados alineados son ideales: un ajuste que accede a un fragmento de memoria contiguo comenzando en una dirección de memoria alineada. Para maximizar el rendimiento de la memoria global, es importante organizar las operaciones de la memoria para que estén alineadas y fusionadas.
En general, debe optimizar la eficiencia de las transacciones de memoria: utilice la menor cantidad de transacciones para atender la cantidad máxima de solicitudes de memoria.

Global Memory Reads.
En un SM, los datos se canalizan a través de una de las siguientes tres rutas de caché / búfer, según el tipo de memoria del dispositivo al que se hace referencia:
Cache L1 / L2.
Cache constante.
Caché de sólo lectura.
La caché L1 / L2 es la ruta predeterminada. Pasar datos a través de las otras dos rutas requiere una gestión explícita por parte de la aplicación, pero puede conducir a una mejora del rendimiento dependiendo de los patrones de acceso utilizados.
En las GPU Fermi (capacidad de procesamiento 2.x) y GPU Kepler K40 o posteriores (capacidad de procesamiento 3.5 y superiores), el almacenamiento en caché L1 de las cargas de memoria global se pueden habilitar o deshabilitar con indicadores del compilador.
Con la caché L1 deshabilitada, todas las solicitudes de carga a la memoria global van directamente a la caché L2; cuando ocurre una falla L2, las solicitudes son atendidas por DRAM.

Cached Loads.
Las operaciones de carga en caché pasan a través de la caché L1 y son atendidas por transacciones de memoria del dispositivo con la granularidad de una línea de caché L1, 128 bytes. Las cargas almacenadas en caché se pueden clasificar como alineadas / desalineadas y fusionadas / no fusionadas.

Uncached Loads.
Las cargas no almacenadas en caché no pasan a través de la caché L1 y se realizan con la granularidad de los segmentos de memoria (32 bytes) y no con las líneas de caché (128 bytes).

Example of Misaligned Reads.
Debido a que los patrones de acceso a menudo están determinados por el algoritmo que implementa una aplicación, puede resultar complicado fusionar cargas de memoria para algunas aplicaciones. Sin embargo, existen técnicas que pueden ayudar a alinear los accesos a la memoria de las aplicaciones en la mayoría de los casos.

Read-Only Cache.
La caché de sólo lectura estaba originalmente reservada para uso de cargas de memoria de textura.
La granularidad de las cargas a través del caché de sólo lectura es de 32 bytes.
Hay dos formas de dirigir las lecturas de memoria a través del caché de sólo lectura:
Usando la función _Idg.
Usar un calificador de declaración en el puntero que se está desreferenciando.

Memory Load Access Patterns.
Hay dos tipos de cargas de memoria:
Carga en cache (cache L1 habilitada).
Carga sin cache (cache L1 deshabilitada).
El patrón de acceso para cargas de memoria se puede caracterizar por las siguientes combinaciones:
En caché versus sin caché: La carga se almacena en caché si la caché L1 está habilitada.
Alineado versus desalineado: La carga está alineada si la primera dirección de un acceso a memoria es múltiplo de 32 bytes.
Fusionado versus no fusionado: La carga se fusiona si un warp accede a una porción contigua de datos.

Difference Between CPU L1 Cache And GPU L1 Cache.
La caché L1 de la CPU está optimizada para la localidad espacial y temporal. La caché GPU L1 está diseñada para localidad espacial pero no temporal.

Global Memory Writes.
Las operaciones del almacén de memoria son relativamente simples. La caché L1 no se utiliza para operaciones de almacenamiento en las GPU Fermi o Kepler; las operaciones de almacenamiento solo se almacenan en caché en la caché L2 antes de enviarse a la memoria del dispositivo. Los almacenes se realizan con una granularidad de segmento de 32 bytes.

Array of Structures versus Structure of Arrays.
Como programador de C, debe estar familiarizado con dos formas de organizar datos: una matriz de estructuras (AoS) y una estructura de matrices (SoA).
El almacenamiento de datos en formato SoA aprovecha al máximo el ancho de banda de la memoria de la GPU.

AoS Versus SoA.
Muchos paradigmas de programación paralela, en particular los paradigmas de estilo SIMD, prefieren SoA. En la programación CUDA C, también se suele preferir SoA porque los elementos de datos están predispuestos para un acceso combinado eficiente a la memoria global, ya que los elementos de datos del mismo campo a los que haría referencia la misma operación de memoria se almacenan uno al lado del otro.

Performance Tuning.
Hay dos objetivos por alcanzar al optimizar la utilización del ancho de banda de la memoria del dispositivo:
Accesos a memoria alineados y fusionados que reducen el ancho de banda desperdiciado.
Suficientes operaciones de memoria simultáneas para ocultar la latencia de la memoria.
Recuerde que maximizar los accesos simultáneos a la memoria se logra mediante:
Incrementar el número de operaciones de memoria independientes realizadas dentro de cada hilo.
Experimentar con la configuración de ejecución de un lanzamiento de kernel para exponer suficiente paralelismo en cada SM.

Maximizing Bandwidth Utilization.
Hay dos factores principales que influyen en el rendimiento de las operaciones de memoria del dispositivo:
Uso eficiente de los bytes que se mueven entre la DRAM del dispositivo y la memoria en chip SM: para evitar desperdiciar el ancho de banda de la memoria del dispositivo, los patrones de acceso a la memoria deben estar alineados y fusionados.
Número de operaciones de memoria en vuelo simultáneamente: Maximizar el número de operaciones de memoria en vuelo es posible mediante 1) desenrollando, generando más accesos a memoria independientes por subproceso, o 2) modificando la configuración de ejecución de un lanzamiento de kernel para exponer más paralelismo con cada SM.

What Bandwidth Can A Kernel Achieve?
Al analizar el rendimiento del kernel, es importante centrarse en la latencia de la memoria, el tiempo para satisfacer una solicitud de memoria individual, y el ancho de banda de la memoria, la velocidad a la que un SM puede acceder a la memoria del dispositivo, medida en bytes por unidad de tiempo.

Memory Bandwidth.
La mayoría de los núcleos son muy sensibles al ancho de banda de la memoria, es decir, están limitados al ancho de banda de la memoria. Hay dos tipos de ancho de banda:
Ancho de banda teórico.
Ancho de banda efectivo.
El ancho de banda teórico es el ancho de banda máximo absoluto que se puede lograr con el hardware disponible. Para un Fermi M2090 con ECC desactivado, el ancho de banda de memoria teórico máximo del dispositivo es 177,6 GB/s.

Matrix Transpose Problem.
La transpuesta de matrices es un problema básico en álgebra lineal. Tomar la transpuesta de una matriz implica intercambiar cada fila con la columna correspondiente.
El acceso zancado es el peor patrón de acceso a la memoria para el rendimiento de las GPU. Sin embargo, es inevitable en operaciones de transposición de matrices.

Setting An Upper And Lower Performance Bound For Transpose Kernels.
Antes de implementar el núcleo de transposición matricial, primero puede crear dos núcleos de copia para calcular los límites superior e inferior aproximados para todos los núcleos de transposición:
Copie la matriz cargando y almacenando filas (límite superior). Esto simula realizar la misma cantidad de operaciones de memoria que la transposición pero solo con accesos fusionados.
Copie la matriz cargando y almacenando columnas (límite inferior). Esto simula realizar la misma cantidad de operaciones de memoria que la transposición pero solo con accesos escalonados.

Matrix Addition With Unified Memory.
Para simplificar la administración de espacios de memoria separados del host y del dispositivo y para ayudar a que este programa CUDA sea más legible y fácil de mantener, puede aplicar las siguientes revisiones a la función principal de sima de matrices usando la Memoria Unificada:
Reemplace las asignaciones de memoria del host y del dispositivo con asignaciones de memoria administradas para eliminar punteros duplicados.
Elimine todas las copias de memoria explícitas.
Debido a que el lanzamiento del kernel es asíncrono con respecto al host y ya no es necesaria una llamada de bloqueo a cudaMemcpy con la memoria administrada, debe sincronizar explícitamente en el lado del host antes de acceder directamente a la salida del kernel.
Si está realizando pruebas en un sistema con más de un dispositivo GPU, se requiere un paso adicional para la aplicación administrada. Debido a que las asignaciones de memoria administrada son visibles para todos los dispositivos en un sistema, querrás restringir qué dispositivo es visible para tu aplicación para que la memoria administrada se asigne a un solo dispositivo.

Conclusión.
En estos capítulos se vieron los modelos de ejecución de CUDA además del uso y administración de la memoria, conociendo de forma en la que CUDA funciona y usa la memoria se puede optimizar a su máximo potencial el software que se está desarrollando.
