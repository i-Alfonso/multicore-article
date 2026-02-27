#!/bin/bash
# =============================================================================
# MINI PROYECTO 1 — Vectorización de un Kernel Numérico
# Operador Prewitt — Detección de Bordes en Imágenes PGM
# Maestría en Sistemas Distribuidos — BUAP
# Dr. Mario Rossainz López — Programación en Plataformas Multicore
# =============================================================================
# USO:
#   chmod +x setup_proyecto_prewitt.sh
#   ./setup_proyecto_prewitt.sh
#
# QUÉ HACE ESTE SCRIPT (automáticamente):
#   PASO 0  → Activa Intel oneAPI (source setvars.sh)
#   PASO 1  → Crea estructura de carpetas
#   PASO 2  → Genera input.pgm (256x256, P5 binario)
#   PASO 3  → Crea KernelEscalar.cpp
#   PASO 4  → Crea KernelAuto.cpp
#   PASO 5  → Crea KernelGuiada.cpp
#   PASO 6  → Crea KernelExplicita.cpp
#   PASO 7  → Compila las 4 versiones con icpx (genera .optrpt)
#   PASO 8  → Verifica los reportes de vectorización
#   PASO 9  → Ejecuta las 4 versiones y muestra métricas
#   PASO 10 → Verifica imágenes de salida
#   FINAL   → Muestra comandos de Advisor para que los corras manualmente
# =============================================================================

set -e
VERDE='\033[0;32m'; AMARILLO='\033[1;33m'; ROJO='\033[0;31m'
CYAN='\033[0;36m';  BLANCO='\033[1;37m';   NC='\033[0m'
sep()   { echo -e "${CYAN}=================================================================${NC}"; }
titulo(){ sep; echo -e "${BLANCO}  $1${NC}"; sep; }
ok()    { echo -e "${VERDE}  [OK] $1${NC}"; }
info()  { echo -e "${AMARILLO}  --> $1${NC}"; }
err()   { echo -e "${ROJO}  [ERROR] $1${NC}"; exit 1; }

# =============================================================================
# PASO 0 — ACTIVAR INTEL ONEAPI
# =============================================================================
titulo "PASO 0 — Activando Intel oneAPI"

ONEAPI="/opt/intel/oneapi/setvars.sh"
[ -f "$ONEAPI" ] || err "No se encontro $ONEAPI — Verifica instalacion de oneAPI"
info "source $ONEAPI --force"
source "$ONEAPI" --force 2>/dev/null || true
command -v icpx &>/dev/null || err "icpx no disponible despues de activar oneAPI"
ok "Compilador activo: $(icpx --version 2>&1 | head -1)"

# =============================================================================
# PASO 1 — ESTRUCTURA DE CARPETAS
# =============================================================================
titulo "PASO 1 — Creando estructura de carpetas"

BASE="$HOME/proyecto_prewitt"
mkdir -p "$BASE/Escalar" "$BASE/Auto" "$BASE/Guiada" "$BASE/Explicita"
mkdir -p "$BASE/latex-informe/Figuras"
ok "Creado: $BASE/{Escalar,Auto,Guiada,Explicita,latex-informe/Figuras}"

# =============================================================================
# PASO 2 — GENERAR IMAGEN DE ENTRADA input.pgm (256x256, formato P5)
# =============================================================================
titulo "PASO 2 — Generando imagen de entrada input.pgm (256x256)"

cat > "$BASE/generar_imagen.cpp" << 'GENEOF'
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
GENEOF

cd "$BASE"
# Usar imagen de entrada ya existente (input.pgm real del usuario)
[ -f "$BASE/input.pgm" ] || err "No se encontro input.pgm en $BASE"
cp input.pgm Escalar/ && cp input.pgm Auto/
cp input.pgm Guiada/  && cp input.pgm Explicita/
ok "input.pgm copiado a las 4 carpetas ($(ls -lh input.pgm | awk '{print $5}'))"

# =============================================================================
# PASO 3 — CREAR KernelEscalar.cpp
# =============================================================================
titulo "PASO 3 — Creando KernelEscalar.cpp"

cat > "$BASE/Escalar/KernelEscalar.cpp" << 'ESCEOF'
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
ESCEOF

ok "KernelEscalar.cpp creado ($(wc -l < $BASE/Escalar/KernelEscalar.cpp) lineas)"

# =============================================================================
# PASO 4 — CREAR KernelAuto.cpp
# =============================================================================
titulo "PASO 4 — Creando KernelAuto.cpp"

cat > "$BASE/Auto/KernelAuto.cpp" << 'AUTOEOF'
// =============================================================================
// VERSION AUTOMATICA — Operador Prewitt, vectorizacion por el compilador
//
// Compilacion:
//   icpx -g -O3 -march=native -std=c++17 -fno-inline
//        -qopt-report=max -qopt-report-phase=vec
//        -o auto KernelAuto.cpp
//
// Con -O3 -march=native el compilador detecta el hardware disponible y
// genera instrucciones SIMD automaticamente. En CPUs modernos puede emitir:
//   - AVX2:    registros YMM, 256 bits, 8 floats por iteracion
//   - AVX-512: registros ZMM, 512 bits, 16 floats por iteracion
// El .optrpt reporta exactamente que ocurrio, el vector length elegido y
// el speedup estimado del kernel.
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
// KERNEL AUTOMATICO — magnitud del gradiente
//
// El codigo fuente es identico al escalar. La diferencia es exclusivamente
// en los flags de compilacion: -O3 -march=native activan la vectorizacion
// automatica. El compilador decide el ancho vectorial optimo para el CPU.
// __restrict garantiza ausencia de aliasing entre los punteros.
// ---------------------------------------------------------------------------
void kernel_auto(const float* __restrict Gx,
                 const float* __restrict Gy,
                 float* __restrict Mag, int N) {
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
    kernel_auto(Gx, Gy, Mag, TOT);
    auto t1=chrono::high_resolution_clock::now();
    double dt=chrono::duration<double>(t1-t0).count();

    imprimir_metricas("Automatica (Prewitt, icpx -O3 -march=native)", dt, TOT, FPE);
    escribir_pgm("output_auto.pgm", Mag, W, H);
    cout << "Guardada: output_auto.pgm\n";

    _mm_free(Gx); _mm_free(Gy); _mm_free(Mag);
    return 0;
}
AUTOEOF

ok "KernelAuto.cpp creado ($(wc -l < $BASE/Auto/KernelAuto.cpp) lineas)"

# =============================================================================
# PASO 5 — CREAR KernelGuiada.cpp
# =============================================================================
titulo "PASO 5 — Creando KernelGuiada.cpp"

cat > "$BASE/Guiada/KernelGuiada.cpp" << 'GUIEOF'
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
GUIEOF

ok "KernelGuiada.cpp creado ($(wc -l < $BASE/Guiada/KernelGuiada.cpp) lineas)"

# =============================================================================
# PASO 6 — CREAR KernelExplicita.cpp
# =============================================================================
titulo "PASO 6 — Creando KernelExplicita.cpp"

cat > "$BASE/Explicita/KernelExplicita.cpp" << 'EXPEOF'
// =============================================================================
// VERSION EXPLICITA AVX2 — Operador Prewitt, intrinsecos _mm256_*
//
// Compilacion:
//   icpx -g -O3 -march=native -std=c++17 -fno-inline
//        -qopt-report=max -qopt-report-phase=vec
//        -o explicita KernelExplicita.cpp
//
// En lugar de dejar que el compilador decida como vectorizar, aqui el
// programador escribe directamente las instrucciones vectoriales AVX2
// usando funciones intrínsecas _mm256_* (registros YMM de 256 bits).
// Cada instruccion YMM procesa 8 floats de 32 bits simultaneamente.
//
// El .optrpt reportara el loop de intrinsecos como "not vectorized by
// compiler" (correcto: ya esta vectorizado manualmente). El loop residual
// puede ser revectorizado por el compilador con mayor ancho vectorial.
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
// KERNEL EXPLICITO — intrinsecos AVX2 (registros YMM, 256 bits)
//
// Loop principal (i += 8): procesa 8 floats por iteracion
//   _mm256_loadu_ps  → carga 8 floats de Gx[i..i+7] o Gy[i..i+7]
//   _mm256_mul_ps    → Gx[i]^2 y Gy[i]^2 para 8 elementos en paralelo
//   _mm256_add_ps    → Gx^2 + Gy^2 para 8 elementos en paralelo
//   _mm256_sqrt_ps   → sqrt para 8 elementos en paralelo
//   _mm256_storeu_ps → escribe 8 resultados en Mag[i..i+7]
//
// Loop residual (escalar): maneja elementos cuando N no es multiplo de 8
// ---------------------------------------------------------------------------
void kernel_explicito(const float* __restrict Gx,
                      const float* __restrict Gy,
                      float* __restrict Mag, int N) {
    int i = 0;

    // --- Loop principal AVX2: 8 floats/iteracion con registros YMM ---
    for (; i <= N-8; i += 8) {
        __m256 vx  = _mm256_loadu_ps(&Gx[i]);   // cargar Gx[i..i+7]
        __m256 vy  = _mm256_loadu_ps(&Gy[i]);   // cargar Gy[i..i+7]
        __m256 vx2 = _mm256_mul_ps(vx, vx);     // Gx^2 x8
        __m256 vy2 = _mm256_mul_ps(vy, vy);     // Gy^2 x8
        __m256 sum = _mm256_add_ps(vx2, vy2);   // Gx^2+Gy^2 x8
        __m256 res = _mm256_sqrt_ps(sum);        // sqrt(Gx^2+Gy^2) x8
        _mm256_storeu_ps(&Mag[i], res);          // guardar Mag[i..i+7]
    }

    // --- Loop residual escalar: elementos restantes (N % 8) ---
    for (; i < N; i++)
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
    kernel_explicito(Gx, Gy, Mag, TOT);
    auto t1=chrono::high_resolution_clock::now();
    double dt=chrono::duration<double>(t1-t0).count();

    imprimir_metricas("Explicita (Prewitt, AVX2 _mm256_*)", dt, TOT, FPE);
    escribir_pgm("output_explicita.pgm", Mag, W, H);
    cout << "Guardada: output_explicita.pgm\n";

    _mm_free(Gx); _mm_free(Gy); _mm_free(Mag);
    return 0;
}
EXPEOF

ok "KernelExplicita.cpp creado ($(wc -l < $BASE/Explicita/KernelExplicita.cpp) lineas)"

# =============================================================================
# PASO 7 — COMPILAR LAS 4 VERSIONES CON icpx
# =============================================================================
titulo "PASO 7 — Compilando con icpx (genera .optrpt automaticamente)"

info "[1/4] ESCALAR: icpx -g -O2 -no-vec ..."
cd "$BASE/Escalar"
icpx -g -O2 -no-vec -std=c++17 -fno-inline \
     -qopt-report=max -qopt-report-phase=vec \
     -o escalar KernelEscalar.cpp 2>&1 | tee compile_escalar.log
ok "Escalar compilado → $(ls -lh escalar | awk '{print $5}')"

info "[2/4] AUTOMATICA: icpx -g -O3 -march=native ..."
cd "$BASE/Auto"
icpx -g -O3 -march=native -std=c++17 -fno-inline \
     -qopt-report=max -qopt-report-phase=vec \
     -o auto KernelAuto.cpp 2>&1 | tee compile_auto.log
ok "Automatica compilada → $(ls -lh auto | awk '{print $5}')"

info "[3/4] GUIADA: icpx -g -O3 -march=native -qopenmp ..."
cd "$BASE/Guiada"
icpx -g -O3 -march=native -qopenmp -std=c++17 -fno-inline \
     -qopt-report=max -qopt-report-phase=vec \
     -o guiada KernelGuiada.cpp 2>&1 | tee compile_guiada.log
ok "Guiada compilada → $(ls -lh guiada | awk '{print $5}')"

info "[4/4] EXPLICITA: icpx -g -O3 -march=native ..."
cd "$BASE/Explicita"
icpx -g -O3 -march=native -std=c++17 -fno-inline \
     -qopt-report=max -qopt-report-phase=vec \
     -o explicita KernelExplicita.cpp 2>&1 | tee compile_explicita.log
ok "Explicita compilada → $(ls -lh explicita | awk '{print $5}')"

# =============================================================================
# PASO 8 — VERIFICAR REPORTES DE VECTORIZACION
# =============================================================================
titulo "PASO 8 — Verificando .optrpt (reportes de vectorizacion)"

echo ""
echo -e "${AMARILLO}[Escalar] kernel_escalar — esperado: NOT vectorized${NC}"
grep -E "not vectorized|NOT vectorized|VECTORIZED" \
     "$BASE/Escalar/KernelEscalar.optrpt" 2>/dev/null | head -6

echo ""
echo -e "${AMARILLO}[Auto] kernel_auto — esperado: LOOP WAS VECTORIZED, length 16${NC}"
grep -E "LOOP WAS VECTORIZED|vector length|speedup" \
     "$BASE/Auto/KernelAuto.optrpt" 2>/dev/null | head -8

echo ""
echo -e "${AMARILLO}[Guiada] kernel_guiado — esperado: SIMD LOOP WAS VECTORIZED, length 8${NC}"
grep -E "SIMD LOOP WAS VECTORIZED|LOOP WAS VECTORIZED|vector length|speedup" \
     "$BASE/Guiada/KernelGuiada.optrpt" 2>/dev/null | head -8

echo ""
echo -e "${AMARILLO}[Explicita] kernel_explicito — loop principal: not vectorized (ya manual)${NC}"
grep -E "LOOP WAS VECTORIZED|not vectorized|vector length" \
     "$BASE/Explicita/KernelExplicita.optrpt" 2>/dev/null | head -10

# =============================================================================
# PASO 9 — EJECUTAR LAS 4 VERSIONES
# =============================================================================
titulo "PASO 9 — Ejecutando las 4 versiones (ANOTAR tiempos y GFLOPS)"

echo ""
echo -e "${BLANCO}============================================================${NC}"
echo -e "${BLANCO}  ESCALAR${NC}"
echo -e "${BLANCO}============================================================${NC}"
cd "$BASE/Escalar" && ./escalar

echo -e "${BLANCO}============================================================${NC}"
echo -e "${BLANCO}  AUTOMATICA${NC}"
echo -e "${BLANCO}============================================================${NC}"
cd "$BASE/Auto" && ./auto

echo -e "${BLANCO}============================================================${NC}"
echo -e "${BLANCO}  GUIADA${NC}"
echo -e "${BLANCO}============================================================${NC}"
cd "$BASE/Guiada" && ./guiada

echo -e "${BLANCO}============================================================${NC}"
echo -e "${BLANCO}  EXPLICITA${NC}"
echo -e "${BLANCO}============================================================${NC}"
cd "$BASE/Explicita" && ./explicita

# =============================================================================
# PASO 10 — VERIFICAR IMAGENES DE SALIDA
# =============================================================================
titulo "PASO 10 — Verificando imagenes de salida PGM"

echo ""
for img in "$BASE/Escalar/output_escalar.pgm" \
           "$BASE/Auto/output_auto.pgm" \
           "$BASE/Guiada/output_guiada.pgm" \
           "$BASE/Explicita/output_explicita.pgm"; do
    if [ -f "$img" ]; then
        ok "$(basename $img)  [$(ls -lh $img | awk '{print $5}')]"
    else
        echo -e "${ROJO}  No encontrado: $img${NC}"
    fi
done

# =============================================================================
# INSTRUCCIONES FINALES PARA ADVISOR
# =============================================================================
titulo "SCRIPT COMPLETADO — COMANDOS PARA INTEL ADVISOR"

echo -e "${BLANCO}"
cat << 'INSTRUCCIONES'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PASO A — Correr Advisor CLI (survey + tripcounts con FLOP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cd ~/proyecto_prewitt/Escalar
  advisor --collect=survey --project-dir=./adv_escalar -- ./escalar
  advisor --collect=tripcounts --flop --project-dir=./adv_escalar -- ./escalar

  cd ~/proyecto_prewitt/Auto
  advisor --collect=survey --project-dir=./adv_auto -- ./auto
  advisor --collect=tripcounts --flop --project-dir=./adv_auto -- ./auto

  cd ~/proyecto_prewitt/Guiada
  advisor --collect=survey --project-dir=./adv_guiada -- ./guiada
  advisor --collect=tripcounts --flop --project-dir=./adv_guiada -- ./guiada

  cd ~/proyecto_prewitt/Explicita
  advisor --collect=survey --project-dir=./adv_explicita -- ./explicita
  advisor --collect=tripcounts --flop --project-dir=./adv_explicita -- ./explicita

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PASO B — Abrir GUI de Advisor (una por version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  advisor-gui ~/proyecto_prewitt/Escalar/adv_escalar &
  advisor-gui ~/proyecto_prewitt/Auto/adv_auto &
  advisor-gui ~/proyecto_prewitt/Guiada/adv_guiada &
  advisor-gui ~/proyecto_prewitt/Explicita/adv_explicita &

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PASO C — Capturas de pantalla (3 por version = 12 PNG total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  fig_survey_escalar.png    → Pestana Survey, seccion Program Metrics
  fig_code_escalar.png      → Pestana Code Analytics
  fig_roofline_escalar.png  → Pestana Roofline (grafico completo)

  (repetir para auto / guiada / explicita)

  Copiar los 12 PNG a:
    ~/proyecto_prewitt/latex-informe/Figuras/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PASO D — Datos a anotar y pasar a Claude para el LaTeX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Version     Tiempo(s)  GFLOPS  AI-Advisor  VecLen
  ─────────────────────────────────────────────────
  Escalar     ___        ___     ___         —
  Automatica  ___        ___     ___         ___
  Guiada      ___        ___     ___         ___
  Explicita   ___        ___     ___         8

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCCIONES
echo -e "${NC}"
