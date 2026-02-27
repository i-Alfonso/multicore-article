// =============================================================================
// VERSION GUIADA — Operador Prewitt, vectorizacion con #pragma omp simd
//
// Compilacion:
//   icpx -g -O3 -march=native -qopenmp -std=c++17 -fno-inline
//        -qopt-report=max -qopt-report-phase=vec
//        -o guiada KernelGuiada.cpp
//
// El pragma #pragma omp simd es una directiva del estandar OpenMP 4.0 que
// ordena explicitamente al compilador vectorizar el loop que lo sigue.
// La clausula simdlen(8) fija el ancho vectorial a 8 elementos float,
// correspondiente a registros YMM de 256 bits (AVX2).
//
// Diferencia clave vs version automatica:
//   - Automatica: el compilador elige libremente el ancho (puede ser 16)
//   - Guiada: el programador ordena y controla el ancho (fijado a 8)
//
// El .optrpt reportara: "SIMD LOOP WAS VECTORIZED"
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

void escribir_pgm(const string& fname, float* buf, int w, int h) {
    ofstream f(fname, ios::binary);
    if (!f.is_open()) return;
    f << "P5\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w*h; i++) {
        unsigned char v = (unsigned char)min(255.0f, max(0.0f, buf[i]));
        f.write((char*)&v, 1);
    }
}

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
// KERNEL GUIADO — vectorizacion dirigida por pragma OpenMP SIMD
//
// #pragma omp simd simdlen(8):
//   Ordena vectorizar con ancho de 8 floats (256 bits = AVX2).
//   El programador garantiza que no hay dependencias entre iteraciones.
//   __restrict confirma que los punteros no se solapan en memoria.
// ---------------------------------------------------------------------------
void kernel_guiado(const float* __restrict Gx,
                   const float* __restrict Gy,
                   float* __restrict Mag, int N) {
    #pragma omp simd simdlen(8)
    for (int i = 0; i < N; i++)
        Mag[i] = sqrtf(Gx[i]*Gx[i] + Gy[i]*Gy[i]);
}

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

    const int ITER=800, N=W*H, TOT=N*ITER, FPE=4;
    float* Gx  = (float*)_mm_malloc(TOT*sizeof(float), 32);
    float* Gy  = (float*)_mm_malloc(TOT*sizeof(float), 32);
    float* Mag = (float*)_mm_malloc(TOT*sizeof(float), 32);
    for (int i=0;i<TOT;i++){Gx[i]=0;Gy[i]=0;Mag[i]=0;}

    calcular_gradientes_prewitt(img, Gx, Gy, W, H);
    for (int k=1;k<ITER;k++){
        int o=k*N;
        for(int i=0;i<N;i++){Gx[o+i]=Gx[i];Gy[o+i]=Gy[i];}
    }

    auto t0=chrono::high_resolution_clock::now();
    kernel_guiado(Gx, Gy, Mag, TOT);
    auto t1=chrono::high_resolution_clock::now();
    double dt=chrono::duration<double>(t1-t0).count();

    imprimir_metricas("Guiada (Prewitt, #pragma omp simd simdlen(8))", dt, TOT, FPE);
    escribir_pgm("output_guiada.pgm", Mag, W, H);
    cout << "Guardada: output_guiada.pgm\n";

    _mm_free(Gx); _mm_free(Gy); _mm_free(Mag);
    return 0;
}
