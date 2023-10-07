//Codigo para mostrar una imagen desde C++ usando OpenCV ademas de modificaciones con solor maps

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