// Genera una imagen PGM 256x256 con:
//   - Patron de ajedrez (bloques 32x32, intensidades 200 y 50)
//   - Rectangulo central 96x96 con contraste maximo (255/10)
// Produce bordes bien definidos para verificar la deteccion con Prewitt.
// Formato de salida: PGM binario P5
#include <fstream>
#include <cstring>
#include <iostream>
using namespace std;
int main() {
    const int W = 256, H = 256;
    unsigned char img[H][W];
    memset(img, 0, sizeof(img));
    // Patron ajedrez 8x8 bloques de 32 pixeles
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (((x/32) + (y/32)) % 2 == 0) ? 200 : 50;
    // Rectangulo central con contraste maximo
    for (int y = 80; y < 176; y++)
        for (int x = 80; x < 176; x++)
            img[y][x] = (img[y][x] > 100) ? 255 : 10;
    // Escribir PGM binario P5
    ofstream f("input.pgm", ios::binary);
    if (!f.is_open()) { cerr << "Error creando input.pgm\n"; return 1; }
    f << "P5\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; y++)
        f.write((char*)img[y], W);
    f.close();
    cout << "input.pgm generado: " << W << "x" << H << " P5 binario\n";
    return 0;
}
