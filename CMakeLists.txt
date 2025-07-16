cmake_minimum_required(VERSION 3.28)
project(RenderingTester CXX) # Nome del progetto in CLion

set(CMAKE_CXX_STANDARD 20) # Usa C++20

# Trova e configura OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui) # Aggiungi altri componenti di OpenCV se ti servono

if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "Errore: OpenCV non trovato. Assicurati che sia installato e configurato.")
endif()

# Aggiungi tutti i file sorgente per il tuo eseguibile principale
# Ora solo test.cpp (che contiene il main) e renderer.cpp
add_executable(RendererTester # Nome dell'eseguibile compilato
        main.cpp
        renderer.cpp
        renderer.h
)

# Collega le directory degli header di OpenCV all'eseguibile
target_include_directories(RendererTester PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

# Collega le librerie di OpenCV all'eseguibile
target_link_libraries(RendererTester PRIVATE
        ${OpenCV_LIBS}
)

# Trova e configura OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message(STATUS "OpenMP trovato e abilitato per il progetto.")
    # Applica le flag di compilazione per OpenMP al target (per le direttive #pragma)
    target_compile_options(RendererTester PRIVATE ${OpenMP_CXX_FLAGS})
    # Collega le librerie di OpenMP al target
    target_link_libraries(RendererTester PRIVATE OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "Errore: OpenMP non trovato. Assicurati che il tuo compilatore lo supporti.")
endif()