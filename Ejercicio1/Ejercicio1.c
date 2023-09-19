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