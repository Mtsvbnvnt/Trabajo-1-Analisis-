import numpy as np
import cmath
import time
import matplotlib.pyplot as plt
import pandas as pd

# SCC Manual con bucles for explícitos
def scc_manual(x, h):
    N = len(x)
    M = len(h)
    L = N + M - 1
    x_padded = np.concatenate([x, np.zeros(L - N, dtype=np.complex_)])
    h_padded = np.concatenate([h, np.zeros(L - M, dtype=np.complex_)])
    y = np.zeros(L, dtype=np.complex_)

    for n in range(L):
        for k in range(M):
            if n - k >= 0:
                y[n] += h_padded[k] * x_padded[n - k]
    return y

# FFT recursiva
def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# IFFT recursiva
def ifft(X):
    N = len(X)
    if N <= 1:
        return X
    even = ifft(X[0::2])
    odd = ifft(X[1::2])
    T = [cmath.exp(2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [(even[k] + T[k]) / 2 for k in range(N // 2)] + [(even[k] - T[k]) / 2 for k in range(N // 2)]

# Convolución con FFT
def convolucion_fft(x, h):
    N = len(x) + len(h) - 1
    N_fft = 1 << (N - 1).bit_length()
    x_padded = np.concatenate([x, np.zeros(N_fft - len(x), dtype=np.complex_)])
    h_padded = np.concatenate([h, np.zeros(N_fft - len(h), dtype=np.complex_)])
    X = fft(x_padded)
    H = fft(h_padded)
    Y = [X[i] * H[i] for i in range(N_fft)]
    y = ifft(Y)
    return np.real(y[:N])

# Variables globales
Ns_global = [2**i for i in range(1, 14)]
tiempos_scc_global = []
tiempos_fft_global = []
scc_done = False
fft_done = False

# Opción 1
def ejecutar_scc():
    global tiempos_scc_global, scc_done
    tiempos_scc_global = []
    print("\n--- Ejecutando SCC Manual ---")
    for N in Ns_global:
        x = np.random.uniform(-1, 1, N).astype(np.complex_)
        h = np.random.uniform(-1, 1, N).astype(np.complex_)
        start = time.process_time()
        scc_manual(x, h)
        end = time.process_time()
        tiempos_scc_global.append(end - start)
        print(f"N = {N}, Tiempo SCC = {end - start:.6f} seg")
    scc_done = True

# Opción 2
def ejecutar_fft():
    global tiempos_fft_global, fft_done
    tiempos_fft_global = []
    print("\n--- Ejecutando Convolución vía FFT ---")
    for N in Ns_global:
        x = np.random.uniform(-1, 1, N).astype(np.complex_)
        h = np.random.uniform(-1, 1, N).astype(np.complex_)
        start = time.process_time()
        convolucion_fft(x, h)
        end = time.process_time()
        tiempos_fft_global.append(end - start)
        print(f"N = {N}, Tiempo FFT = {end - start:.6f} seg")
    fft_done = True

# Opción 3
def comparar_resultados():
    print("\n--- Comparación de tiempos (tabla + gráfico) ---")
    df = pd.DataFrame({
        "Tamaño N": Ns_global,
        "SCC Manual": tiempos_scc_global,
        "Convolución FFT": tiempos_fft_global
    })
    print(df.to_string(index=False))
    df.to_csv("resultados_convolucion.csv", index=False)
    print("Archivo CSV guardado como 'resultados_convolucion.csv'")

    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(Ns_global, tiempos_scc_global, label='SCC Manual', marker='o')
    plt.plot(Ns_global, tiempos_fft_global, label='Convolución FFT', marker='x')
    plt.xscale('log', base=2)  # Eje X logarítmico
    plt.yscale('log')  # Eje Y logarítmico
    # Y queda lineal
    plt.xlabel('Tamaño de la señal (N)')
    plt.ylabel('Tiempo de ejecución (segundos)')
    plt.title('Comparación de Tiempos de Ejecución')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafico_comparacion.png")
    print("Gráfico guardado como 'grafico_comparacion.png'")
    plt.show()


# Menú
if __name__ == "__main__":
    while True:
        print("\nSeleccione una opción:")
        print("1. Ejecutar SCC Manual (con bucles for)")
        print("2. Ejecutar Convolución vía FFT")
        print("3. Comparar resultados y guardar CSV/PNG")
        print("4. Salir")

        opcion = input("Opción: ").strip()

        if opcion == "1":
            ejecutar_scc()
        elif opcion == "2":
            ejecutar_fft()
        elif opcion == "3":
            if scc_done and fft_done:
                comparar_resultados()
            else:
                print("Debes ejecutar primero las opciones 1 y 2.")
        elif opcion == "4":
            print("Saliendo...")
            break
        else:
            print("Opción no válida.")
