CUDA Cores:
Son unos procesadores paralelos que se encargan de procesar todos los datos que entran y salen de la GPU, realizando calculos graficos cuyo resultado los ve el usuario final.
Se encuentran dentro de la GPU y sus tareas frecuentes son renderizar objetos 3D, dibujar modelos, comprender y resolver la iluminacion y sombreado de una escena, etc.

Threads:
Son conocidos como sub-procesos o hilos y se definen como una secuencia de instrucciones las cuales el sistema operativo puede programar para su ejecucion.
A diferencia de un proceso, los threads son entidades mucho mas pequeñas, lo cual los hace faciles de gestionar y es la unidad mas pequeña a la cual un procesador puede asignar tiempo.
A diferencia de los procesos, los cuales estan dentro del sistema operativo, los threads se encuentran dentro de los procesos.
Un thread se crea, ejecuta y finaliza dentro de un proceso.

Blocks and Grids:
Al lanzarse un kernel, CUDA genera un grid (cuadricula) de threads (hilos) que son organizados en una gerarquia de forma tridimencional:
- Cada grid es organizado en un array de blocks de threads.
- Cada block puede contener hasta 1024 threads.
- El numero de threads en un block es definido en la variable blockDim.
- Las dimensiones de thread blocks deben ser multiplos de 32.

A grandes rasgos es la forma en la que se separan y organizan los threads de un kernel.

Bibliografia:
“Qué son los Nvidia CUDA Cores y cuál es su importancia”. Profesional Review. Accedido el 4 de octubre de 2023. [En línea]. Disponible: https://www.profesionalreview.com/2018/10/09/que-son-nvidia-cuda-core/
“Threads y Procesos”. CódigoFacilito. Accedido el 4 de octubre de 2023. [En línea]. Disponible: https://codigofacilito.com/articulos/threads-procesos
"CSC 447: Parallel Programming for Multi-Core and Cluster Systems". CUDA Thread Scheduling. Accedido el 4 de octubre de 2023. [En línea]. Disponible: http://harmanani.github.io/classes/csc447/Notes/Lecture15.pdf
