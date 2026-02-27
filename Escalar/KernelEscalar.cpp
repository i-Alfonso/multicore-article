// =============================================================================
// VERSION ESCALAR — Operador Prewitt, sin vectorizacion
//
// Compilacion:
//   icpx -g -O2 -no-vec -std=c++17 -fno-inline
//        -qopt-report=max -qopt-report-phase=vec
//        -o escalar KernelEscalar.cpp
//
// El flag -no-vec prohibe explicitamente la vectorizacion automatica.
// Esto garantiza una linea base escalar pura para comparar con las
// versiones vectorizadas. El .optrpt confirmara que el kernel
// kernel_escalar NO fue vectorizado.
// =============================================================================
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <immintrin.h>
#include <algorithm>
using namespace std;

// ---------------------------------------------------------------------------
// Lectura de imagen PGM — soporta P2 (ASCII) y P5 (binario)
// ---------------------------------------------------------------------------
void leer_pgm(const string& fname, vector<float>& buf, int& w, int& h) {
    ifstream f(fname, ios::binary);
    if (!f.is_open()) { cerr << "Error abriendo " << fname << "\n"; return; }
    string hdr; f >> hdr;
    if (hdr != "P2" && hdr != "P5") { cerr << "PGM no valido\n"; return; }
    char c;
    while (f >> ws && (c = f.peek()) == '#') { string t; getline(f, t); }
    int mv; f >> w >> h >> mv;
    buf.resize(w * h);
    if (hdr == "P2") {
        for (int i = 0; i < w*h; i++) { int p; f >> p; buf[i] = (float)p; }
    } else {
        f.get();
        vector<unsigned char> t(w*h);
        f.read((char*)t.data(), w*h);
        for (int i = 0; i < w*h; i++) buf[i] = (float)t[i];
    }
}

// ---------------------------------------------------------------------------
// Escritura de imagen PGM formato P5 binario
// ---------------------------------------------------------------------------
void escribir_pgm(const string& fname, float* buf, int w, int h) {
    ofstream f(fname, ios::binary);
    if (!f.is_open()) return;
    f << "P5\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w*h; i++) {
        unsigned char v = (unsigned char)min(255.0f, max(0.0f, buf[i]));
        f.write((char*)&v, 1);
    }
}

// ---------------------------------------------------------------------------
// Preprocesamiento: gradientes con operador PREWITT
//
// A diferencia de Sobel, Prewitt usa coeficientes uniformes (todos +/-1)
// sin ponderar la fila/columna central con +/-2.
//
//  Mascara Kx (detecta bordes verticales):
//    [-1  0 +1]
//    [-1  0 +1]
//    [-1  0 +1]
//
//  Mascara Ky (detecta bordes horizontales):
//    [+1 +1 +1]
//    [ 0  0  0]
//    [-1 -1 -1]
//
// Solo se procesan pixeles interiores (evitando el borde de la imagen).
// ---------------------------------------------------------------------------
void calcular_gradientes_prewitt(const vector<float>& img,
                                  float* Gx, float* Gy, int w, int h) {
    const int Kx[3][3] = {{-1,0,1},{-1,0,1},{-1,0,1}};
    const int Ky[3][3] = {{ 1,1,1},{ 0,0,0},{-1,-1,-1}};
    for (int y = 1; y < h-1; y++)
        for (int x = 1; x < w-1; x++) {
            float sx = 0, sy = 0;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++) {
                    float p = img[(y+ky)*w+(x+kx)];
                    sx += p * Kx[ky+1][kx+1];
                    sy += p * Ky[ky+1][kx+1];
                }
            Gx[y*w+x] = sx;
            Gy[y*w+x] = sy;
        }
}

// ---------------------------------------------------------------------------
// KERNEL ESCALAR — magnitud del gradiente (este es el loop que Advisor mide)
//
//   Mag[i] = sqrt( Gx[i]^2 + Gy[i]^2 )
//
// Operaciones por elemento: 2 mul + 1 add + 1 sqrt = 4 FLOPs
// El flag -no-vec garantiza que este loop se ejecute escalar puro.
// __restrict informa que los punteros no se solapan en memoria.
// ---------------------------------------------------------------------------
void kernel_escalar(const float* __restrict Gx,
                    const float* __restrict Gy,
                    float* __restrict Mag, int N) {
    for (int i = 0; i < N; i++)
        Mag[i] = sqrtf(Gx[i]*Gx[i] + Gy[i]*Gy[i]);
}

// ---------------------------------------------------------------------------
// Imprime tabla de metricas de rendimiento en terminal
// ---------------------------------------------------------------------------
void imprimir_metricas(const string& ver, double t, long long N, int fpe) {
    double gflops = ((double)N * fpe / t) / 1.0e9;
    double gbs    = ((double)N * 3 * sizeof(float) / t) / 1.0e9;
    double ai     = (double)fpe / (3.0 * sizeof(float));
    cout << "\n====================================================\n";
    cout << "  Version        : " << ver    << "\n";
    cout << "  Tiempo         : " << t      << " s\n";
    cout << "  GFLOPS         : " << gflops << "\n";
    cout << "  Ancho de banda : " << gbs    << " GB/s\n";
    cout << "  AI teorica     : " << ai     << " FLOP/Byte\n";
    cout << "  N total        : " << N      << " elementos\n";
    cout << "====================================================\n\n";
}

int main() {
    int W, H;
    vector<float> img;
    leer_pgm("input.pgm", img, W, H);
    if (img.empty()) return -1;
    cout << "Imagen cargada: " << W << "x" << H << " pixeles\n";

    const int ITER = 800;              // Iteraciones para estabilizar Advisor
    const int N    = W * H;            // 65,536 elementos por imagen
    const int TOT  = N * ITER;         // 52,428,800 elementos en total
    const int FPE  = 4;                // FLOPs por elemento

    // Memoria alineada a 32 bytes (requerimiento de instrucciones AVX2)
    float* Gx  = (float*)_mm_malloc(TOT*sizeof(float), 32);
    float* Gy  = (float*)_mm_malloc(TOT*sizeof(float), 32);
    float* Mag = (float*)_mm_malloc(TOT*sizeof(float), 32);
    for (int i = 0; i < TOT; i++) { Gx[i]=0; Gy[i]=0; Mag[i]=0; }

    // Calcular gradientes Prewitt sobre la imagen real
    calcular_gradientes_prewitt(img, Gx, Gy, W, H);

    // Replicar 800 veces para estabilizar las metricas de Advisor
    for (int k = 1; k < ITER; k++) {
        int o = k*N;
        for (int i = 0; i < N; i++) { Gx[o+i]=Gx[i]; Gy[o+i]=Gy[i]; }
    }

    // Medir tiempo del kernel
    auto t0 = chrono::high_resolution_clock::now();
    kernel_escalar(Gx, Gy, Mag, TOT);
    auto t1 = chrono::high_resolution_clock::now();
    double dt = chrono::duration<double>(t1-t0).count();

    imprimir_metricas("Escalar (Prewitt, -no-vec)", dt, TOT, FPE);

    // Guardar imagen con los bordes detectados (solo primer frame)
    escribir_pgm("output_escalar.pgm", Mag, W, H);
    cout << "Guardada: output_escalar.pgm\n";

    _mm_free(Gx); _mm_free(Gy); _mm_free(Mag);
    return 0;
}
