Actividad de C para multiplicar 2 matrices.
----------------------------------------------------
#include <stdio.h>

int main()
{
    int a[10][10], b[10][10], c[10][10];
    int num = 0, aux = 0;
    
    //Dar valor a las matrices
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++) {
            num = num + 1;
            a[i][j] = num;
            b[i][j] = num + 1;
        }
    }

    //Multiplicar las matrices
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++) {
            aux += a[i][j] * b[j][i];
            
            c[i][j] = aux;
        }
    }
    
    //Mostrar la matriz
    for (int i = 0; i < 10; i++){
        
        for (int j = 0; j < 10; j++) {
            printf("%d ", c[i][j]);
            
        }
        
        printf("\n");
    }
    
    return 0;
}
------------------------------------------------------
Se crearon 3 matrices, 2 para ser multiplicadas y 1 para guardar los resultados.
Se les dieron valores a ambas matrices.
Se realizo la multiplicacion de matrices y se guardo el resultado en la tercera.
Se imprimio la tercera matriz.
