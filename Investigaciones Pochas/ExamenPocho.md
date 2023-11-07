Programacion de paralelismo - Examen parcial 2

En las columnas por orden de izquierda a derecha especifican:
-El tipo de actividad que se esta realizando, cada tipo tiene divididos sus procesos y funciones que realizan.
-El porcentaje de tiempo que tomo hacer cada funcion segun el dispositivo.
-El tiempo que tomo hacer la funcion.
-Las veces o llamadas de la funcion.
-El promedio de tiempo que tomo realizar las funciones.
-El minimo de tiempo que necesito para realizar la funcion.
-El maximo de tiempo que necesito para realizar la funcion.
-Nombre de la funcion que se realizo.

Todo esto ordenado de arriba a abajo tomando en cuenta la cantidad de tiempo que llevo realizar la accion dividido para cada tipo de actividad (GPU activities y API calls).
----------------------------------------------------------------------------------------------------------------------------------------------------------
==1011== NVPROF is profiling process 1011, command: ./simpleMathAoS
==1011== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1011== Profiling application: ./simpleMathAoS
==1011== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   80.19%  23.304ms         2  11.652ms  8.6804ms  14.623ms  [CUDA memcpy DtoH]
                   18.05%  5.2457ms         1  5.2457ms  5.2457ms  5.2457ms  [CUDA memcpy HtoD]
                    0.88%  256.10us         1  256.10us  256.10us  256.10us  warmup(innerStruct*, innerStruct*, int)
                    0.88%  255.85us         1  255.85us  255.85us  255.85us  testInnerStruct(innerStruct*, innerStruct*, int)
      API calls:   88.83%  566.38ms         2  283.19ms  376.90us  566.01ms  cudaMalloc
                    5.61%  35.742ms         1  35.742ms  35.742ms  35.742ms  cudaDeviceReset
                    4.90%  31.267ms         3  10.422ms  6.6946ms  15.576ms  cudaMemcpy
                    0.34%  2.1562ms         1  2.1562ms  2.1562ms  2.1562ms  cuDeviceGetPCIBusId
                    0.19%  1.2200ms         2  610.00us  446.90us  773.10us  cudaFree
                    0.11%  669.90us         2  334.95us  334.90us  335.00us  cudaDeviceSynchronize
                    0.03%  161.60us         2  80.800us  63.500us  98.100us  cudaLaunchKernel
                    0.00%  14.600us       101     144ns     100ns  1.3000us  cuDeviceGetAttribute
                    0.00%  6.1000us         1  6.1000us  6.1000us  6.1000us  cudaSetDevice
                    0.00%  5.8000us         2  2.9000us  2.6000us  3.2000us  cudaGetLastError
                    0.00%  4.7000us         1  4.7000us  4.7000us  4.7000us  cudaGetDeviceProperties
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

En este codigo se uso un array de estructuras (AoS: Array of Structure) para almacenar datos en el dispositivo y ver su desempeño por medio de la GPU.
Lo mas destacable es el uso del cudaMalloc que fue llamado solo 2 veces y necesito 566.38ms para ser completado, esta consiste en asignar bytes de tamaño de memoria lineal en el dispositivo y regresa un puntero a la memoria asignada, en esta se puede asignar cualquier valor y no se borra.
Y la funcion cuDeviceGetAttribute que aunque fue llamada 101 veces solo demoro 14.6us representando menos del 1% del tiempo de las API calls, esta funcion regresa informacion del dispositivo.

-[CUDA memcpy DtoH]: Hace copias de la memoria desde el device hacia el host - represento el 80.19% del tiempo de ejecucion con 23.304ms - llamado 2 veces.
-[CUDA memcpy HtoD]: Hace copias de la memoria desde el host hacia el device - represento el 18.05% del tiempo de ejecucion con 5.2457ms - llamado 1 vez.
-warmup(innerStruct*, innerStruct*, int): Funcion global que guarda valores temporales X y Y para una matriz - represento el 0.88% del tiempo de ejecucion con 256.10us - llamado 1 vez.
-testInnerStruct(innerStruct*, innerStruct*, int): Funcion global ingresa valores a una matriz - represento el 0.88% del tiempo de ejecucion con 255.85us - llamado 1 vez.

-cudaMalloc: Asigna bytes de tamaño de memoria lineal en el dispositivo y devuelve en *devPtr un puntero a la memoria asignada - represento el 88.83% del tiempo de ejecucion con 566.38ms - llamado 2 veces.
-cudaDeviceReset: Destruye y limpia todos los recursos asociados con el dispositivo actual en el proceso actual - represento el 5.61% del tiempo de ejecucion con 35.742ms - llamado 1 vez.
-cudaMemcpy: Copia datos entre el host y el dispositivo - represento el 4.90% del tiempo de ejecucion con 31.267ms - llamado 3 veces.
-cuDeviceGetPCIBusId: Devuelve la configuracion de la memoria compartida para el dispositivo actual - represento el 0.34% del tiempo de ejecucion con 2.1562ms - llamado 1 vez.
-cudaFree: Libera memoria en el dispositivo - represento el 0.19% del tiempo de ejecucion con 1.2200ms - llamado 2 veces.
-cudaDeviceSynchronize: Espera a que finaliza el dispositivo - represento el 0.11% del tiempo de ejecucion con 669.90us - llamado 2 veces.
-cudaLaunchKernel: Inicia una funcion del dispositivo - represento el 0.03% del tiempo de ejecucion con 161.60us - llamado 2 veces.
-cuDeviceGetAttribute: Regresa informacion del dispositivo - represento el 0.00% del tiempo de ejecucion con 14.600us - llamado 101 veces.
-cudaSetDevice: Configura el dispositivo que se utilizara para las ejecuciones de la GPU - represento el 0.00% del tiempo de ejecucion con 5.8000us - llamado 1 vez.
-cudaGetLastError: Devuelve el ultimo error de una llamada en tiempo de ejecucion - represento el 0.00% del tiempo de ejecucion con 6.1000us - llamado 1 vez.
-cudaGetDeviceProperties: Devuelve informacion sobre el diuspositivo informatico - represento el 0.00% del tiempo de ejecucion con 4.7000us - llamado 1 vez.
-cuDeviceGetCount: Devuelve el numero de dispositivos con capacidad informatica - represento el 0.00% del tiempo de ejecucion con 1.4000us - llamado 3 veces.
-cuDeviceGet: Devuelve un identificador a un dispositivo informatico - represento el 0.00% del tiempo de ejecucion con 800ns - llamado 2 veces.
-cuDeviceGetName: Devuelve una cadena de identificacion para el dispositivo - represento el 0.00% del tiempo de ejecucion con 600ns - llamado 1 vez.
-cuDeviceTotalMem: Devuelve la cantidad total de memoria del dispositivo - represento el 0.00% del tiempo de ejecucion con 400ns - llamado 1 vez.
-cuDeviceGetUuid: Devuelve un UUID para el dispositivo - represento el 0.00% del tiempo de ejecucion con 200ns - llamado 1 vez.
----------------------------------------------------------------------------------------------------------------------------------------------
==1027== NVPROF is profiling process 1027, command: ./simpleMathSoA
==1027== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1027== Profiling application: ./simpleMathSoA
==1027== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   73.35%  12.215ms         2  6.1076ms  3.7599ms  8.4554ms  [CUDA memcpy DtoH]
                   23.58%  3.9265ms         1  3.9265ms  3.9265ms  3.9265ms  [CUDA memcpy HtoD]
                    1.54%  256.42us         1  256.42us  256.42us  256.42us  warmup2(InnerArray*, InnerArray*, int)
                    1.54%  256.03us         1  256.03us  256.03us  256.03us  testInnerArray(InnerArray*, InnerArray*, int)
      API calls:   90.98%  584.89ms         2  292.45ms  380.00us  584.51ms  cudaMalloc
                    5.47%  35.165ms         1  35.165ms  35.165ms  35.165ms  cudaDeviceReset
                    2.89%  18.564ms         3  6.1881ms  3.9129ms  9.2690ms  cudaMemcpy
                    0.39%  2.4897ms         1  2.4897ms  2.4897ms  2.4897ms  cuDeviceGetPCIBusId
                    0.15%  981.80us         2  490.90us  359.80us  622.00us  cudaFree
                    0.11%  682.20us         2  341.10us  302.90us  379.30us  cudaDeviceSynchronize
                    0.01%  94.200us         2  47.100us  43.700us  50.500us  cudaLaunchKernel
                    0.00%  16.500us       101     163ns     100ns  1.4000us  cuDeviceGetAttribute
                    0.00%  5.9000us         1  5.9000us  5.9000us  5.9000us  cudaSetDevice
                    0.00%  5.1000us         1  5.1000us  5.1000us  5.1000us  cudaGetDeviceProperties
                    0.00%  4.7000us         2  2.3500us  2.3000us  2.4000us  cudaGetLastError
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%  1.2000us         1  1.2000us  1.2000us  1.2000us  cuDeviceGetName
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

En este codigo se uso una estrucura de matrices (SoA: Structure of Array) para almacenar datos en el dispositivo, parecido al anterior pero ejecutado de diferente manera.
Aunque el tiempo que tomaron hacer ciertas funciones fue bastante similar, en conjunto este metodo si mejoro el tiempo de ejecucion por unos cuantos milisegundos.

-[CUDA memcpy DtoH]: Hace copias de la memoria desde el device hacia el host - represento el 73.35% del tiempo de ejecucion con 12.215ms - llamado 2 veces.
-[CUDA memcpy HtoD]: Hace copias de la memoria desde el host hacia el device - represento el 23.58% del tiempo de ejecucion con 3.9265ms - llamado 1 vez.
-warmup2(InnerArray*, InnerArray*, int): Introduce los datos en sus correspondientes espacios en la matriz - represento el 1.54% del tiempo de ejecucion con 256.42us - llamado 1 vez.
-testInnerArray(InnerArray*, InnerArray*, int): Coloca sus valores en el array - represento el 1.54% del tiempo de ejecucion con 256.03us - llamado 1 vez.

-cudaMalloc: Asigna bytes de tamaño de memoria lineal en el dispositivo y devuelve en *devPtr un puntero a la memoria asignada - represento el 90.98% del tiempo de ejecucion con 584.89ms - llamado 2 veces.
-cudaDeviceReset: Destruye y limpia todos los recursos asociados con el dispositivo actual en el proceso actual - represento el 5.47% del tiempo de ejecucion con 35.165ms - llamado 1 vez.
-cudaMemcpy: Copia datos entre el host y el dispositivo - represento el 2.89% del tiempo de ejecucion con 18.564ms - llamado 3 veces.
-cuDeviceGetPCIBusId: Devuelve la configuracion de la memoria compartida para el dispositivo actual - represento el 0.39% del tiempo de ejecucion con 2.4897ms - llamado 1 vez.
-cudaFree: Libera memoria en el dispositivo - represento el 0.15% del tiempo de ejecucion con 981.80us - llamado 2 veces.
-cudaDeviceSynchronize: Espera a que finaliza el dispositivo - represento el 0.11% del tiempo de ejecucion con 682.20us - llamado 2 veces.
-cudaLaunchKernel: Inicia una funcion del dispositivo - represento el 0.01% del tiempo de ejecucion con 94.200us - llamado 2 veces.
-cuDeviceGetAttribute: Regresa informacion del dispositivo - represento el 0.00% del tiempo de ejecucion con 16.500us - llamado 101 veces.
-cudaSetDevice: Configura el dispositivo que se utilizara para las ejecuciones de la GPU - represento el 0.00% del tiempo de ejecucion con 5.9000us - llamado 1 vez.
-cudaGetDeviceProperties: Devuelve informacion sobre el diuspositivo informatico - represento el 0.00% del tiempo de ejecucion con 5.1000us - llamado 1 vez.
-cudaGetLastError: Devuelve el ultimo error de una llamada en tiempo de ejecucion - represento el 0.00% del tiempo de ejecucion con 4.7000us - llamado 1 vez.
-cuDeviceGetCount: Devuelve el numero de dispositivos con capacidad informatica - represento el 0.00% del tiempo de ejecucion con 1.4000us - llamado 3 veces.
-cuDeviceGetName: Devuelve una cadena de identificacion para el dispositivo - represento el 0.00% del tiempo de ejecucion con 1.2000us - llamado 1 vez.
-cuDeviceGet: Devuelve un identificador a un dispositivo informatico - represento el 0.00% del tiempo de ejecucion con 1.1000us - llamado 2 veces.
-cuDeviceTotalMem: Devuelve la cantidad total de memoria del dispositivo - represento el 0.00% del tiempo de ejecucion con 300ns - llamado 1 vez.
-cuDeviceGetUuid: Devuelve un UUID para el dispositivo - represento el 0.00% del tiempo de ejecucion con 200ns - llamado 1 vez.
--------------------------------------------------------------------------------------------------------------------------------------------------------------
==1049== NVPROF is profiling process 1049, command: ./sumArrayZerocpy
==1049== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1049== Profiling application: ./sumArrayZerocpy
==1049== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   33.33%  3.5200us         1  3.5200us  3.5200us  3.5200us  sumArraysZeroCopy(float*, float*, float*, int)
                   22.73%  2.4000us         2  1.2000us  1.1840us  1.2160us  [CUDA memcpy DtoH]
                   22.12%  2.3360us         1  2.3360us  2.3360us  2.3360us  sumArrays(float*, float*, float*, int)
                   21.82%  2.3040us         2  1.1520us     864ns  1.4400us  [CUDA memcpy HtoD]
      API calls:   94.24%  583.14ms         3  194.38ms  1.8000us  583.14ms  cudaMalloc
                    5.09%  31.475ms         1  31.475ms  31.475ms  31.475ms  cudaDeviceReset
                    0.35%  2.1756ms         1  2.1756ms  2.1756ms  2.1756ms  cuDeviceGetPCIBusId
                    0.16%  988.60us         2  494.30us  3.8000us  984.80us  cudaHostAlloc
                    0.06%  368.90us         2  184.45us  4.5000us  364.40us  cudaFreeHost
                    0.06%  358.00us         4  89.500us  33.100us  129.40us  cudaMemcpy
                    0.04%  218.20us         3  72.733us  2.5000us  208.10us  cudaFree
                    0.01%  60.300us         2  30.150us  28.600us  31.700us  cudaLaunchKernel
                    0.00%  14.900us       101     147ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  6.2000us         1  6.2000us  6.2000us  6.2000us  cudaSetDevice
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cudaGetDeviceProperties
                    0.00%  2.1000us         2  1.0500us     600ns  1.5000us  cudaHostGetDevicePointer
                    0.00%  1.6000us         3     533ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

En este codigo se muestra el uso de la memoria zero-copy para quitar la necesidad de hacer una operacion memcpy entre el host y el dispositivo.
Con esta implementacion las actividades del GPU quedan reducidas a microsegundos aunque sume arrays porque se elimino la necesidad de usar operaciones memcpy.

-sumArraysZeroCopy(float*, float*, float*, int): Suma dos arrays y los introduce en otro array - represento el 33.33% del tiempo de ejecucion con 3.5200us - llamado 1 vez.
-[CUDA memcpy DtoH]: Hace copias de la memoria desde el device hacia el host - represento el 22.73% del tiempo de ejecucion con 2.4000us - llamado 2 veces.
-sumArrays(float*, float*, float*, int): Suma dos arrays y los introduce en otro array - represento el 22.12% del tiempo de ejecucion con 2.3360us - llamado 1 vez.
-[CUDA memcpy HtoD]: Hace copias de la memoria desde el host hacia el device - represento el 21.82% del tiempo de ejecucion con 2.3040us - llamado 1 vez.

La neta de aqui en adelante solo incluire las funciones que sean nuevas porque muchas se repiten y sus tiempos de ejecucion son muy similares juas juas.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
==1071== NVPROF is profiling process 1071, command: ./sumMatrixGPUManaged
==1071== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1071== Profiling application: ./sumMatrixGPUManaged
==1071== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  12.948ms         2  6.4741ms  288.67us  12.660ms  sumMatrixGPU(float*, float*, float*, int, int)
      API calls:   91.39%  815.38ms         4  203.85ms  27.532ms  731.17ms  cudaMallocManaged
                    3.45%  30.801ms         1  30.801ms  30.801ms  30.801ms  cudaDeviceReset
                    3.31%  29.569ms         4  7.3922ms  7.2484ms  7.4490ms  cudaFree
                    1.52%  13.583ms         1  13.583ms  13.583ms  13.583ms  cudaDeviceSynchronize
                    0.24%  2.1681ms         1  2.1681ms  2.1681ms  2.1681ms  cuDeviceGetPCIBusId
                    0.07%  644.20us         2  322.10us  11.200us  633.00us  cudaLaunchKernel
                    0.00%  14.100us       101     139ns     100ns     900ns  cuDeviceGetAttribute
                    0.00%  5.8000us         1  5.8000us  5.8000us  5.8000us  cudaSetDevice
                    0.00%  4.0000us         1  4.0000us  4.0000us  4.0000us  cudaGetDeviceProperties
                    0.00%  1.2000us         3     400ns     100ns     900ns  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cudaGetLastError
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

En este codigo se demuestra el uso de la memoria CUDA para hacer una suma de matrices.
En este ejemplo a diferencia de otros se observa que la GPU solo se encarga de realizar una funcion que es la suma de matrices.
Otro detalle interesante es que en vez de usar el cudaMalloc como otros codigos, este usa cudaMallocManaged que asigna memoria que sera administrada automaticamente por el sistema de memoria unificada.

-sumMatrixGPU(float*, float*, float*, int, int): Se realiza una suma de matrices bidimensional (grid 2D y block 2D) - represento el 100.00% del tiempo de ejecucion con 12.948ms - llamado 2 veces.

-cudaMallocManaged: Asigna memoria que sera administrada automaticamente por el sistema de memoria unificada - represento el 91.39% del tiempo de ejecucion con 815.38ms - llamado 4 veces.
-----------------------------------------------------------------------------------------------------------------------------------------------------
==1089== NVPROF is profiling process 1089, command: ./sumMatrixGPUManual
==1089== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1089== Profiling application: ./sumMatrixGPUManual
==1089== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.52%  27.101ms         2  13.550ms  8.3698ms  18.731ms  [CUDA memcpy HtoD]
                   30.63%  12.669ms         1  12.669ms  12.669ms  12.669ms  [CUDA memcpy DtoH]
                    2.69%  1.1118ms         2  555.89us  288.73us  823.04us  sumMatrixGPU(float*, float*, float*, int, int)
                    1.16%  479.42us         2  239.71us  238.91us  240.51us  [CUDA memset]
      API calls:   87.57%  607.17ms         3  202.39ms  713.10us  605.72ms  cudaMalloc
                    6.50%  45.038ms         3  15.013ms  8.6183ms  23.545ms  cudaMemcpy
                    5.26%  36.474ms         1  36.474ms  36.474ms  36.474ms  cudaDeviceReset
                    0.33%  2.2576ms         1  2.2576ms  2.2576ms  2.2576ms  cuDeviceGetPCIBusId
                    0.19%  1.3256ms         3  441.87us  223.90us  799.30us  cudaFree
                    0.13%  929.30us         1  929.30us  929.30us  929.30us  cudaDeviceSynchronize
                    0.01%  62.700us         2  31.350us  24.300us  38.400us  cudaMemset
                    0.01%  62.500us         2  31.250us  28.200us  34.300us  cudaLaunchKernel
                    0.00%  15.600us       101     154ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  7.3000us         1  7.3000us  7.3000us  7.3000us  cudaSetDevice
                    0.00%  7.1000us         1  7.1000us  7.1000us  7.1000us  cudaGetDeviceProperties
                    0.00%  1.3000us         3     433ns     200ns     900ns  cuDeviceGetCount
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%  1.0000us         2     500ns     200ns     800ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cudaGetLastError
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

Este codigo demuestra el uso de la transferencia de memoria CUDA explicita para hacer una suma de matrices, este se parece al anterior pero se diferencia en que el anterior se centraba en eliminar todas las transferencias de memoria explicitas y en este se hace "a mano".
Aqui la GPU tiene mas actividades por hacer como en codigos anteiores y se vuelve a usar el cudaMalloc en vez del cudaMallocManaged, el tiempo de ejecucion de este codigo no se diferencia mucho del anterior pero es un poco mas rapido.

-sumMatrixGPU(float*, float*, float*, int, int): Se realiza una suma de matrices bidimensional (grid 2D y block 2D) - represento el 2.69% del tiempo de ejecucion con 1.1118ms - llamado 2 veces.
-[CUDA memset]: Inicializa o establece la memoria del dispositivo en un valor - represento el 1.16% del tiempo de ejecucion con 479.42us - llamado 2 veces.
------------------------------------------------------------------------------------------------------------------------------------------------------------
==1111== NVPROF is profiling process 1111, command: ./transpose
==1111== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1111== Profiling application: ./transpose
==1111== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   86.82%  1.9853ms         1  1.9853ms  1.9853ms  1.9853ms  [CUDA memcpy HtoD]
                    6.62%  151.49us         1  151.49us  151.49us  151.49us  copyRow(float*, float*, int, int)
                    6.56%  150.02us         1  150.02us  150.02us  150.02us  warmup(float*, float*, int, int)
      API calls:   86.44%  634.10ms         2  317.05ms  434.00us  633.66ms  cudaMalloc
                   12.79%  93.791ms         1  93.791ms  93.791ms  93.791ms  cudaDeviceReset
                    0.32%  2.3634ms         1  2.3634ms  2.3634ms  2.3634ms  cudaMemcpy
                    0.31%  2.2569ms         1  2.2569ms  2.2569ms  2.2569ms  cuDeviceGetPCIBusId
                    0.07%  549.80us         2  274.90us  222.60us  327.20us  cudaFree
                    0.06%  404.50us         2  202.25us  166.80us  237.70us  cudaDeviceSynchronize
                    0.01%  57.000us         2  28.500us  15.400us  41.600us  cudaLaunchKernel
                    0.00%  16.500us       101     163ns     100ns  1.2000us  cuDeviceGetAttribute
                    0.00%  5.4000us         1  5.4000us  5.4000us  5.4000us  cudaSetDevice
                    0.00%  5.0000us         1  5.0000us  5.0000us  5.0000us  cudaGetDeviceProperties
                    0.00%  1.4000us         3     466ns     100ns  1.1000us  cuDeviceGetCount
                    0.00%  1.3000us         2     650ns     600ns     700ns  cudaGetLastError
                    0.00%  1.0000us         2     500ns     200ns     800ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

En este codigo se optimizan varios patrones de acceso a memoria aplicados a un nucleo de transposicion de matriz.
Una vez las actividades de la GPU muestran un tiempo de ejecucion muy pequeño a comparacion de las API calls aunque en las actividades de la GPU se encuentren las "funciones principales" ya que en este codigo se centra en mejorar los patrones de acceso de memoria se usan mas que nada funciones de la misma API clalls.

-copyRow(float*, float*, int, int): Accede y copia los datos en las filas del kernel - represento el 6.62% del tiempo de ejecucion con 151.49us - llamado 1 vez.
--------------------------------------------------------------------------------------------------------------------------------------------------------------
==1127== NVPROF is profiling process 1127, command: ./writeSegment
==1127== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1127== Profiling application: ./writeSegment
==1127== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.98%  2.1129ms         3  704.29us  518.98us  921.45us  [CUDA memcpy DtoH]
                   29.36%  940.23us         2  470.12us  465.19us  475.04us  [CUDA memcpy HtoD]
                    1.55%  49.504us         1  49.504us  49.504us  49.504us  writeOffset(float*, float*, float*, int, int)
                    1.49%  47.712us         1  47.712us  47.712us  47.712us  warmup(float*, float*, float*, int, int)
                    0.91%  29.120us         1  29.120us  29.120us  29.120us  writeOffsetUnroll2(float*, float*, float*, int, int)
                    0.72%  23.072us         1  23.072us  23.072us  23.072us  writeOffsetUnroll4(float*, float*, float*, int, int)
      API calls:   92.61%  579.23ms         3  193.08ms  301.40us  578.59ms  cudaMalloc
                    6.01%  37.576ms         1  37.576ms  37.576ms  37.576ms  cudaDeviceReset
                    0.83%  5.1802ms         5  1.0360ms  537.40us  2.0100ms  cudaMemcpy
                    0.34%  2.1550ms         1  2.1550ms  2.1550ms  2.1550ms  cuDeviceGetPCIBusId
                    0.11%  687.50us         3  229.17us  186.80us  276.10us  cudaFree
                    0.06%  399.00us         4  99.750us  72.600us  145.30us  cudaDeviceSynchronize
                    0.04%  225.60us         4  56.400us  20.100us  89.100us  cudaLaunchKernel
                    0.00%  14.400us       101     142ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  4.9000us         1  4.9000us  4.9000us  4.9000us  cudaGetDeviceProperties
                    0.00%  4.1000us         1  4.1000us  4.1000us  4.1000us  cudaSetDevice
                    0.00%  3.3000us         4     825ns     400ns  1.1000us  cudaGetLastError
                    0.00%  1.2000us         3     400ns     200ns     800ns  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

En este codigo se demuestra el impacto de las escrituras desalineadas en el rendimiento al forzar que las escrituras desalineadas se produzcan en un flotante.
En cuestiones de rendimiento y tiempo de ejecucion no se diferencia de la constante mostrada en otros codigos anteriores, este esta mas enfocado a mostrar el impacto de las escrituras desalineadas.

-writeOffset(float*, float*, float*, int, int): Escribe los datos con cierta separacion unos de otros - represento el 1.55% del tiempo de ejecucion con 49.504us - llamado 1 vez.
-writeOffsetUnroll2(float*, float*, float*, int, int): Escribe los datos con cierta separacion unos de otros de forma desenrrollada 2 veces - represento el 0.91% del tiempo de ejecucion con 29.120us - llamado 1 vez.
-writeOffsetUnroll4(float*, float*, float*, int, int): Escribe los datos con cierta separacion unos de otros de forma desenrrollada 4 veces de forma incremental - represento el 0.72% del tiempo de ejecucion con 23.072us - llamado 1 vez.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
==935== NVPROF is profiling process 935, command: ./memTransfer
==935== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==935== Profiling application: ./memTransfer
==935== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.15%  2.1117ms         1  2.1117ms  2.1117ms  2.1117ms  [CUDA memcpy HtoD]
                   47.85%  1.9374ms         1  1.9374ms  1.9374ms  1.9374ms  [CUDA memcpy DtoH]
      API calls:   93.74%  577.35ms         1  577.35ms  577.35ms  577.35ms  cudaMalloc
                    5.15%  31.729ms         1  31.729ms  31.729ms  31.729ms  cudaDeviceReset
                    0.71%  4.3856ms         2  2.1928ms  2.1784ms  2.2072ms  cudaMemcpy
                    0.34%  2.0994ms         1  2.0994ms  2.0994ms  2.0994ms  cuDeviceGetPCIBusId
                    0.05%  306.30us         1  306.30us  306.30us  306.30us  cudaFree
                    0.00%  14.700us       101     145ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  8.0000us         1  8.0000us  8.0000us  8.0000us  cudaSetDevice
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

En este codigo hace uso de la API de copia de memoria de CUDA para transferir datos hacia y desde el dispositivo, usando ahora cudaMalloc para asignar memoria en la GPU y cudaMemcpy para transferir contenido de la memoria del host a una matriz asignada usando cudaMalloc.
En este caso en particular [CUDA memcpy HtoD] y [CUDA memcpy DtoH] tienen un rendimiento similar de las actividades de la GPU, ambas cercanas al 50%. Y en este codigo se utiliza mas que nada cudaMalloc con casi 94% de la API calls.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
==947== NVPROF is profiling process 947, command: ./pinMemTransfer
==947== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==947== Profiling application: ./pinMemTransfer
==947== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   50.57%  1.3036ms         1  1.3036ms  1.3036ms  1.3036ms  [CUDA memcpy HtoD]
                   49.43%  1.2743ms         1  1.2743ms  1.2743ms  1.2743ms  [CUDA memcpy DtoH]
      API calls:   93.65%  564.84ms         1  564.84ms  564.84ms  564.84ms  cudaHostAlloc
                    5.15%  31.051ms         1  31.051ms  31.051ms  31.051ms  cudaDeviceReset
                    0.45%  2.7319ms         2  1.3660ms  1.3368ms  1.3951ms  cudaMemcpy
                    0.34%  2.0604ms         1  2.0604ms  2.0604ms  2.0604ms  cuDeviceGetPCIBusId
                    0.30%  1.8091ms         1  1.8091ms  1.8091ms  1.8091ms  cudaFreeHost
                    0.06%  342.90us         1  342.90us  342.90us  342.90us  cudaMalloc
                    0.04%  261.00us         1  261.00us  261.00us  261.00us  cudaFree
                    0.00%  15.400us       101     152ns     100ns     900ns  cuDeviceGetAttribute
                    0.00%  7.2000us         1  7.2000us  7.2000us  7.2000us  cudaSetDevice
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cudaGetDeviceProperties
                    0.00%  1.0000us         3     333ns     100ns     700ns  cuDeviceGetCount
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

Este codigo hace practicamente lo mismo que el anterior pero con la diferencia de que la memoria del host se asigna mediante cudaMallocHost para crear una matriz de hosts con pagina bloqueada.
En este caso [CUDA memcpy HtoD] y [CUDA memcpy DtoH] tambien tienen un rendimiento aun mas cercano al 50%, en cuanto a las API calls su rendimiento es bastante similar al anterior codigo.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
==963== NVPROF is profiling process 963, command: ./readSegment
==963== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==963== Profiling application: ./readSegment
==963== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.71%  992.10us         1  992.10us  992.10us  992.10us  [CUDA memcpy DtoH]
                   45.41%  906.47us         2  453.23us  447.23us  459.23us  [CUDA memcpy HtoD]
                    2.48%  49.408us         1  49.408us  49.408us  49.408us  readOffset(float*, float*, float*, int, int)
                    2.40%  48.001us         1  48.001us  48.001us  48.001us  warmup(float*, float*, float*, int, int)
      API calls:   93.88%  603.77ms         3  201.26ms  313.00us  603.14ms  cudaMalloc
                    5.02%  32.299ms         1  32.299ms  32.299ms  32.299ms  cudaDeviceReset
                    0.52%  3.3638ms         3  1.1213ms  585.30us  2.1168ms  cudaMemcpy
                    0.40%  2.5464ms         1  2.5464ms  2.5464ms  2.5464ms  cuDeviceGetPCIBusId
                    0.13%  833.20us         3  277.73us  167.00us  455.50us  cudaFree
                    0.03%  206.30us         2  103.15us  68.900us  137.40us  cudaDeviceSynchronize
                    0.01%  65.800us         2  32.900us  16.900us  48.900us  cudaLaunchKernel
                    0.00%  15.800us       101     156ns     100ns  1.4000us  cuDeviceGetAttribute
                    0.00%  5.5000us         1  5.5000us  5.5000us  5.5000us  cudaSetDevice
                    0.00%  4.9000us         1  4.9000us  4.9000us  4.9000us  cudaGetDeviceProperties
                    0.00%  1.2000us         2     600ns     600ns     600ns  cudaGetLastError
                    0.00%     900ns         3     300ns     100ns     600ns  cuDeviceGetCount
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

Este codigo demuestra el impacto de las lecturas desalineadas en el rendimiento al forzar que se produzcan lecturas desalineadas en un flotante.
En las actividades de la GPU el rendimiento de las funciones no duraron ni milisegundos en realizarse, donde se genera mayor peso en el rendimiento es haciendo el cudaMalloc 3 veces llamado en este codigo.

-readOffset(float*, float*, float*, int, int): Se suman dos valores con offset para introducir el resultado en un array - represento el 2.48% del tiempo de ejecucion con 49.408us - llamado 1 vez.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
==985== NVPROF is profiling process 985, command: ./readSegmentUnroll
==985== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==985== Profiling application: ./readSegmentUnroll
==985== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.13%  2.0672ms         3  689.07us  470.56us  864.49us  [CUDA memcpy DtoH]
                   27.79%  895.65us         2  447.83us  446.53us  449.12us  [CUDA memcpy HtoD]
                    1.94%  62.593us         4  15.648us  15.360us  16.320us  [CUDA memset]
                    1.56%  50.368us         1  50.368us  50.368us  50.368us  readOffsetUnroll4(float*, float*, float*, int, int)
                    1.55%  49.984us         1  49.984us  49.984us  49.984us  readOffset(float*, float*, float*, int, int)
                    1.54%  49.632us         1  49.632us  49.632us  49.632us  readOffsetUnroll2(float*, float*, float*, int, int)
                    1.49%  47.904us         1  47.904us  47.904us  47.904us  warmup(float*, float*, float*, int, int)
      API calls:   93.30%  592.46ms         3  197.49ms  309.10us  591.77ms  cudaMalloc
                    5.46%  34.676ms         1  34.676ms  34.676ms  34.676ms  cudaDeviceReset
                    0.69%  4.4052ms         5  881.04us  498.20us  1.8633ms  cudaMemcpy
                    0.32%  2.0617ms         1  2.0617ms  2.0617ms  2.0617ms  cuDeviceGetPCIBusId
                    0.12%  749.60us         3  249.87us  170.00us  390.60us  cudaFree
                    0.06%  357.30us         4  89.325us  71.700us  130.70us  cudaDeviceSynchronize
                    0.02%  144.90us         4  36.225us  22.500us  52.700us  cudaMemset
                    0.01%  91.300us         4  22.825us  9.4000us  47.700us  cudaLaunchKernel
                    0.00%  14.600us       101     144ns     100ns  1.3000us  cuDeviceGetAttribute
                    0.00%  6.9000us         1  6.9000us  6.9000us  6.9000us  cudaGetDeviceProperties
                    0.00%  5.7000us         1  5.7000us  5.7000us  5.7000us  cudaSetDevice
                    0.00%  2.4000us         4     600ns     500ns     700ns  cudaGetLastError
                    0.00%  1.5000us         2     750ns     300ns  1.2000us  cuDeviceGet
                    0.00%  1.2000us         3     400ns     100ns     900ns  cuDeviceGetCount
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

En este codigo se realiza el mismo objetivo que el codigo anterior pero en este tambien se incluyen los nucleos que reducen el impacto en el rendimiento de las lecturas desalineadas mediante el desenrollado.
En las actividades de la GPU aunque se realizan varias funciones multiples veces estas no representan una gran carga para el rendimiento ya que duran algunos microsegundos, en cuanto a las API calls en lo que recae mas peso es en la ejecucion de cudaMalloc multiples veces.

-readOffsetUnroll4(float*, float*, float*, int, int): Se suman dos valores con offset para introducir el resultado en un array consecutivamente de forma desenrrollada de 0 a 3 (4 veces) - represento el 1.56% del tiempo de ejecucion con 50.368us - llamado 1 vez.
-readOffsetUnroll2(float*, float*, float*, int, int): Se suman dos valores con offset para introducir el resultado en un array consecutivamente de forma desenrrollada de 0 a 1 (2 veces) - represento el 49.632us del tiempo de ejecucion con 50.368us - llamado 1 vez.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
