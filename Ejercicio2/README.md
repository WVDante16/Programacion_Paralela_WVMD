Mostrar una imagen usando C++

Imagen digital:
Representacion bidimensional de una imagen a partir de una matriz numerica, frecuentemente en binario.
Dependiendo de si la solucion de la imagen es estatica o dinamica, puede trararse de una imagen matricial (mapa de bits) o de un grafico vectorial.

Pixel:
Es la menor unidad homogenea en color que forma parte de una imagen digital.

Matriz:
Conjunto bidimensional de numeros o simbolos distribuidos de forma rectangular, en lineas verticales y horizontales, de manera que sus elementos se organizan en filas y columnas.

Codigo:
Codigo para mostrar una imagen desde C++ usando OpenCV ademas de modificaciones con color maps

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <filesystem>

int main() {
    std::string image_path;
    std::cout << "Enter the path to the image: ";
    std::cin >> image_path;

    //If para confirmar si existe una imagen en la direccion asignada
    if (!std::filesystem::exists(image_path)) {
        std::cout << "File does not exist at the specified path" << std::endl;
        return -1;
    }

    //Mat se refiere a la matriz de pixeles
    cv::Mat image = cv::imread(image_path);

    //If para confirmar que el archivo es una imagen
    if (image.empty()) {
        std::cout << "Error loading the image" << std::endl;
    }
    else {
        std::cout << "Image loaded successfully" << std::endl;
    }

    //Mostrar la imagen
    cv::imshow("Image", image);

    //Esperar a presionar una tecla para continuar el programa
    cv::waitKey(0);

    //Imagen en escala de grises
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::imshow("Grayscale image", grayImage);
    
    //Esperar a presionar una tecla para continuar el programa
    cv::waitKey(0);

    //Separar la imagen en tres canales
    cv::Mat rgb[3];
    cv::split(image, rgb);

    //Modificar la imagen con color maps
    cv::Mat redChannel, greenChannel, blueChannel;
    cv::applyColorMap(rgb[0], redChannel, cv::COLORMAP_PINK);
    cv::applyColorMap(rgb[1], greenChannel, cv::COLORMAP_PLASMA);
    cv::applyColorMap(rgb[2], blueChannel, cv::COLORMAP_TWILIGHT);

    //Mostrar las imagenes modificadas en pantalla
    cv::imshow("redChannel", redChannel);
    cv::imshow("greenChannel", greenChannel);
    cv::imshow("blueChannel", blueChannel);

    //Esperar a presionar una tecla para continuar el programa
    cv::waitKey(0);
}

Complejidad temporal:
O(N)
