Checking The Data Layout Of Shared Memory.
Al diseñar sus propios núcleos que utilizan memoria compartida, debe centrarse en los dos concepto siguientes:
Mapeo de elementos de datos entre bancos de memoria.
Mapeo del índice de subprocesos al desplazamiento de memoria compartida.

Square Shared Memory.
Puede utilizar la memoria compartida para almacenar en caché datos globales con dimensiones cuadradas de forma sencilla. La dimensionalidad simple de una matriz cuadrada facilita el cálculo de compensaciones de memoria 1D a partir de índices de subprocesos 2D.
Puede declarar estáticamente una variable de memoria compartida 2D, de la siguiente manera: 
_shared__int tile[N][N];
Debido a que este tile de memoria compartida es cuadrado, puede optar por acceder a él desde un bloque de subprocesos 2D con subprocesos vecinos que acceden a elementos vecinos en la dimensión x o y: 
tile[threadIdx.y][threadIdx.x]
tile[threadIdx.x][threadIdx.y]
Debe prestar atención a cómo se asignan los subprocesos a los bancos de memoria compartida. Recuerde que lo óptimo es tener hilos en el mismo warp accediendo a bancos separados. Los hilos en el mismo warp se pueden identificar mediante valores consecutivos de threadIdx.x. Los elementos de la memoria compartida que pertenecen a diferentes bancos también se almacenan consecutivamente, mediante desplazamiento de palabras.

Accessing Row-Major Versus Column-Major.
Considere un ejemplo en el que se utiliza una cuadrícula con un bloque 2D que contiene 32 hilos en cada dimensión. 
El kernel tiene dos operaciones simples:
Escribir índices de subprocesos globales en una matriz de memoria compartida 2D en orden de fila principal.
Leer esos valores de la memoria compartida en orden de la fila principal y guardarlos en memoria global.
Debido a que los subprocesos en el mismo warp tienen valores threadIdx.x consecutivos y usan threadId.x para indexar la dimensión más interna del tile de la matriz de memoria compartida, este núcleo está libre de conflictos bancarios.
Por otro lado, si intercambia threadIdx.y y threadId.x al asignar datos al mosaico de memoria compartida, los accesos a la memoria de un warp estarán en orden de columna principal.

Writing Row-Major And Reading Column-Major.
El siguiente kernel implementa escrituras en memoria compartida en orden de fila principal y lecturas de memoria compartida en orden de columna principal. La escritura en el tile de memoria compartida en orden de fila principal se implementa colocando la dimensión más interna del índice del subproceso como índice de columna del tile de memoria compartida 2D:
tile[threadIdx.y][threadIdx.x] = idx;
La asignación de valores a la memoria global desde el tile de memoria compartida en el orden de las columnas principales se implementa intercambiando los dos índices de subprocesos al hacer referencia a la memoria compartida:
out[idx] = tile[threadIdx.x][threadIdx.y];

Dynamic Shared Memory.
Puede implementar estos mismos núcleos declarando la memoria compartida dinámicamente. Puede declarar la memoria compartida dinámicamente fuera del kernel para hacerla global al alcance del archivo, o dentro del kernel para restringirla al alcance del kernel. La memoria compartida dinámica debe declararse como una matriz 1D sin tamaño; por lo tanto, es necesario calcular los índices de subprocesos 2D.
Escribe en la memoria compartida en origen de fila principal utilizando el row_idx calculado de la siguiente manera: tile[row_idx] = row_idx;
Usando la sincronización adecuada después de que se haya llenado el tile de memoria compartida, luego lo lee en el orden de las columnas principales y lo asigna a la memoria global de la siguiente manera: out[row_idx] = tile[col_idx];
Debido a que out se almacena en la memoria global y los subprocesos están organizados en orden de fila principal dentro de un bloque de subprocesos, desea escribir en orden de fila principal por coordenada de subproceso para garantizar almacenes fusionados.
El tamaño de la memoria compartida debe especificarse al iniciar el kernel, de la siguiente manera: setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);

Padding Statically Declared Shared Memory.
Como se describe en el segmento “Memory Padding” de este capítulo, el relleno de matrices es una forma de evitar conflictos bancarios. Para hacer padding simplemente agregue una columna a la asignación de memoria compartida 2D de la siguiente manera: __shared__ int tile[BDIMY][BDIMX+1];
El siguiente kernel es una revisión del kernel setRowReadCol, que informó un conflicto de 16 vías al leer en el orden de las columnas principales. Al rellenar un elemento en cada fila, los elementos de la columna se distribuyen entre diferentes bancos, por lo que tanto las operaciones de lectura como de escritura están libres de conflictos.
__global__ void setRowReadColPad(int *out) {
// static shared memory
__shared__ int tile[BDIMY][BDIMX+IPAD];
// mapping from thread index to global memory offset
unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
// shared memory store operation
tile[threadIdx.y][threadIdx.x] = idx;
// wait for all threads to complete
__syncthreads();
// shared memory load operation
out[idx] = tile[threadIdx.x][threadIdx.y];
}

Padding Dynamically Declared Shared Memory.
Hacer padding a una matriz de memoria compartida declarada dinámicamente es más complejo. Debe omitir un espacio de memoria padded para cada fila al realizar la conversión de índices de subprocesos 2D a índices de memoria 1D, de la siguiente manera:
unsigned int row_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
unsigned int col_idx = threadIdx.x * (blockDim.x + 1) + threadIdx.y;
Debido a que la memoria global utilizada para almacenar datos en el siguiente kernel es más pequeña que la memoria compartida rellena, necesita tres índices: un índice para escrituras de la fila principal en la memoria compartida, un índice para lecturas de columna principal de la memoria compartida y un índice para accesos combinados a la memoria global no rellenada, como se muestra: 
__global__ void setRowReadColDynPad(int *out) {
// dynamic shared memory
extern __shared__ int tile[];
// mapping from thread index to global memory index
unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
unsigned int col_idx = threadIdx.x * (blockDim.x + IPAD) + threadIdx.y;
unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;
// shared memory store operation
tile[row_idx] = g_idx;
// wait for all threads to complete
__syncthreads();
// shared memory load operation
out[g_idx] = tile[col_idx];
}

Rectangular Shared Memory.
La memoria compartida rectangular es un caso más general de memoria compartida 2D, donde el número de filas y columnas de una matriz no son iguales.
__shared__ int tile[Row][Col];
No puede simplemente cambiar las coordenadas del hilo utilizadas para hacer referencia a una matriz rectangular al realizar una operación de transposición, como la implementación de memoria compartida cuadrada. Hacerlo provocaría una infracción de acceso a la memoria al utilizar la memoria compartida rectangular.
Sin pérdida de generalidad, examinará una matriz de memoria compartida rectangular con 32 elementos por fila y 16 elementos por columna. Las dimensiones se definen en las siguientes macros: 
#define BDIMX 32
#define BDIMY 16
El tile de memoria compartida rectangular se asigna de la siguiente manera: __shared__ int tile[BDIMY][BDIMX];
Para simplificar, el kernel se iniciará con una sola cuadrícula y un bloque 2D del mismo tamaño que la matriz de memoria compartida rectangular, de la siguiente manera: 
dim3 block (BDIMX,BDIMY);
dim3 grid (1,1);

Accessing Row-Major Versus Accessing Column-Major.
Debe prestar especial atención a la declaración de la matriz de memoria compartida rectangular en cada núcleo. En el setRowReadRow del kernel, la longitud de la dimensión más interna del tile de la matriz de memoria compartida se establece en la misma dimensión que la dimensión más interna del bloque de subprocesos 2D: __shared__ int tile[BDIMY][BDIMX];
En el setColReadCol del núcleo, la longitud de la dimensión más interna del tile de la matriz de memoria compartida se establece en la misma dimensión que la dimensión más externa del bloque de subprocesos 2D: __shared__ int tile[BDIMX][BDIMY];

Writing Row-Major And Reading Column-Major. 
En esta sección, implementará un kernel que escribe en la memoria compartida en orden de fila principal y lee de la memoria compartida en orden de columna principal utilizando una matriz de memoria compartida rectangular.
El tile de memoria compartida 2D se declara de la siguiente manera: __shared__ int tile[BDIMY][BDIMX];
El kernel tiene tres operaciones de memoria:
Escriba en una fila de memoria compartida con cada warp para evitar conflictos bancarios.
Leer desde una columna de memoria compartida con cada warp para realizar una transposición de matriz.
Escriba en una fila de memoria global desde cada warp con acceso combinado.
El procedimiento para calcular los accesos adecuados a la memoria global y compartida es el siguiente. Primero, el índice de subproceso 2D del subproceso actual se convierte en una ID de subproceso global 1D: unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
Este mapeo 1D de fila principal garantiza que los accesos a la memoria global están fusionados. Debido a que los elementos de datos en la memoria global de salida se transponen, deberá calcular la nueva coordenada en la matriz de transposición, de la siguiente manera:
unsigned int irow = idx / blockDim.y;
unsigned int icol = idx % blockDim.y;
El tile de memoria compartida se inicializa almacenando los ID de subprocesos globales en el tile de memoria compartida 2D de la siguiente manera: tile[threadIdx.y][threadIdx.x] = idx;
En este punto, los datos en la memoria compartida se almacenan linealmente de 0 a BDIMX×BDIMY-1. Dado que cada warp realiza escrituras en las filas principales en la memoria compartida, no hay conflictos bancarios durante la operación de escritura.
Ahora puede acceder a los datos de la memoria compartida que se transpondrán con las coordenadas calculadas anteriormente. Al acceder a la memoria compartida mediante irow e icol intercambiados, puede escribir los datos transpuestos en la memoria global utilizando los ID de subproceso 1D.

Dynamically Declared Shared Memory.
Debido a que la memoria compartida dinámica solo se puede declarar como una matriz 1D, se requiere un nuevo índice para convertir de coordenadas de subprocesos 2D a índices de memoria compartida 1D al escribir por filas y leer por columnas: unsigned int col_idx = icol * blockDim.x + irow;
Debido a que icol corresponde a la dimensión más interna del bloque de subprocesos, esta conversión produce acceso de columna principal a la memoria compartida, lo que resulta en conflictos bancarios.

Padding Statically Declared Shared Memory.
También puede utilizar el relleno de memoria compartida para resolver conflictos bancarios para la memoria compartida rectangular. Sin embargo, para los dispositivos Kepler debes calcular cuántos elementos de relleno se necesitan. Para facilitar la codificación, utilice una macro para definir el número de columnas de relleno agregadas a cada fila: #define NPAD 2
La memoria compartida estática padded se declara de la siguiente manera: __shared__ int tile[BDIMY][BDIMX + NPAD];

Padding Dynamically Declared Shared Memory.
Las técnicas de padding también se pueden aplicar a núcleos de memoria compartida dinámica que utilizan regiones rectangulares de memoria compartida. Debido a que la memoria compartida padded y la memoria global tendrán diferentes tamaños se deben  mantener tres indice por subproceso en el kernel:
row_idx: índice de fila principal de la memoria compartida padded. Usando este índice, un warp puede acceder a una sola fila de la matriz.
col_idx: índice de columna principal de la memoria compartida padded. Usando este índice, un warp puede acceder a una sola columna de matriz.
g_idx: un índice de la memoria global lineal. Usando este índice, un warp puede realizar accesos combinados a la memoria global.
