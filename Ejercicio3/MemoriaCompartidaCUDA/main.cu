#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

//Dimensiones de la matriz
#define X 10
#define Y 10

//Tamaño del padding
#define PADDING_SIZE_X 1

//Tamaños del bloque
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

//Funcion global para aplicar el padding de CPU a GPU
__global__ void ApplyPadding(int *matrix, int *paddedMatrix, int width, int height, int paddingX) {
    //Establecer los valores del indice de los bloques y hilos para ponerlos en cada indice
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    //If que comprueba si los valores de filas y columnas estan fuera de la matriz
    if (col < width && row < height) {
        //Recorrer el valor de la matriz, aplicando el padding
        paddedMatrix[row * (width + paddingX) + col + paddingX] = matrix[row * width + col];
    }
}

//Funcion global para sumar las columnas de la matriz desde la CPU a GPU
__global__ void SumColumns(int *matrix, int *result, int width, int height) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    //If para comprobar si es la matriz padded
    if (col < width) {
        int sum = 0;

        //For para sumar los numeros de la columna guardandolos en la variable
        for (int row = 0; row < height; ++row) {
            sum += matrix[(row) * width + col];
        }

        //Dar nuevo resultado de la suma
        result[col] = sum;
    }
}

int main() {
    int matrix[X][Y]; //Definir valores de la matriz principal antes establecidos
    int paddedMatrix[X + PADDING_SIZE_X][Y]; //Definir matriz con el padding establecido
    int result[Y]; //Variable para guardar el resultado de la suma

    //Variables de la GPU
    int *d_matrix;
    int *d_paddedMatrix;
    int *d_padMatrix;
    int *d_resultSum;

    //Dar valores aleatorios a la matriz
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++) {
            matrix[i][j] = (rand() % 101) + 1; //Numeros aleatorios entre 1 y 100
        }
    }

    //Mostrar la matriz original
    printf("Sin padding:\n");
    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            printf("%d \t", matrix[i][j]);
        }
        printf("\n");
    }

    //Alocacion locochona de memoria de CUDA
    cudaMalloc((void **)&d_matrix, X * Y * sizeof(int));
    cudaMalloc((void **)&d_paddedMatrix, (X + PADDING_SIZE_X) * Y * sizeof(int));
    cudaMalloc((void **)&d_padMatrix, (X + PADDING_SIZE_X) * Y * sizeof(int));
    cudaMalloc((void **)&d_resultSum, Y * sizeof(int));

    //Copiar la matriz a la variable que se envia a la GPU
    cudaMemcpy(d_matrix, matrix, X * Y * sizeof(int), cudaMemcpyHostToDevice);

    //Definir las dimensiones del grid y del block
    dim3 gridDim((Y + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (X + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1);
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

    //Llamar a la funcion para aplicar el padding
    ApplyPadding<<<gridDim, blockDim>>>(d_matrix, d_paddedMatrix, Y, X, PADDING_SIZE_X);

    //Copiar el resultado en la matriz padeada desde el dispositivo al host
    cudaMemcpy(paddedMatrix, d_paddedMatrix, (X + PADDING_SIZE_X) * Y * sizeof(int), cudaMemcpyDeviceToHost);

    //Mostrar matriz con padding
    printf("\nCon padding:\n");
    for (int i = 0; i < X + PADDING_SIZE_X; ++i) {
        for (int j = 0; j < Y; ++j) {
            printf("%d\t", paddedMatrix[i][j]);
        }
        printf("\n");
    }

    //Copiar los valores en la matriz padded
    cudaMemcpy(d_padMatrix, paddedMatrix, (X + PADDING_SIZE_X) * Y * sizeof(int), cudaMemcpyHostToDevice);

    //Llamar a la funcion para sumar las columnas
    SumColumns<<<gridDim, blockDim>>>(d_padMatrix, d_resultSum, Y, X + PADDING_SIZE_X);

    //Copiar el resultado despues de hacer la suma de columnas
    cudaMemcpy(result, d_resultSum, Y * sizeof(int), cudaMemcpyDeviceToHost);

    //Imprimir la matriz con las sumas de columnas
    printf("\n Sumas de la matriz:\n");
    for (int i = 0; i < Y; ++i) {
        printf("%d\t", result[i]);
    }
    printf("\n");

    //Liberar la memoria usada de las variables de GPU
    cudaFree(d_matrix);
    cudaFree(d_paddedMatrix);
    cudaFree(d_padMatrix);
    cudaFree(d_resultSum);

    return 0;
}