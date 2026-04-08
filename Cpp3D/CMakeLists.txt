cmake_minimum_required(VERSION 3.20)
project(PYBEMCS3D VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ── Options ──────────────────────────────────────────────────────────────────
option(USE_OPENMP    "Enable OpenMP parallelism"          ON)
option(USE_VTK       "Enable VTK 3D visualization"        ON)
option(USE_OCCT      "Enable OpenCASCADE STEP import"     ON)

# ── OpenMP ───────────────────────────────────────────────────────────────────
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# ── Eigen (header-only) ─────────────────────────────────────────────────────
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ── Qt6 ─────────────────────────────────────────────────────────────────────
find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGLWidgets)

# ── VTK ─────────────────────────────────────────────────────────────────────
if(USE_VTK)
    find_package(VTK REQUIRED COMPONENTS
        CommonCore CommonDataModel CommonMath
        FiltersSources FiltersGeneral FiltersGeometry
        RenderingCore RenderingOpenGL2 RenderingAnnotation
        InteractionStyle InteractionWidgets
        GUISupportQt
        IOGeometry IOLegacy IOXML
    )
    add_definitions(-DUSE_VTK)
endif()

# ── OpenCASCADE ──────────────────────────────────────────────────────────────
if(USE_OCCT)
    find_package(OpenCASCADE REQUIRED)
    add_definitions(-DUSE_OCCT)
endif()

# ── Source files ─────────────────────────────────────────────────────────────
set(CORE_SOURCES
    src/core/Grid3D.cpp
    src/core/PoissonSolver3D.cpp
    src/core/BorisPusher3D.cpp
    src/core/ParticleManager.cpp
    src/core/CollisionHandler.cpp
    src/core/SputteringModel.cpp
    src/core/ThermalSolver3D.cpp
    src/core/Simulator3D.cpp
)

set(GEOMETRY_SOURCES
    src/geometry/Mesh3D.cpp
    src/geometry/MeshGenerator.cpp
    src/geometry/STEPImporter.cpp
)

set(GUI_SOURCES
    src/gui/MainWindow.cpp
    src/gui/SimulationView3D.cpp
    src/gui/ControlPanel.cpp
    src/gui/MeshingDialog.cpp
    src/gui/GeometryImportDialog.cpp
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    ${CORE_SOURCES}
    ${GEOMETRY_SOURCES}
    ${GUI_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# ── Link libraries ──────────────────────────────────────────────────────────
target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    Qt6::Widgets
    Qt6::OpenGLWidgets
)

if(USE_OPENMP)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(USE_VTK)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${VTK_LIBRARIES})
    vtk_module_autoinit(TARGETS ${PROJECT_NAME} MODULES ${VTK_LIBRARIES})
endif()

if(USE_OCCT)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCASCADE_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE
        TKernel TKMath TKBRep TKTopAlgo TKGeomBase
        TKSTEP TKSTEPBase TKSTEPAttr TKXDESTEP
        TKXSBase TKG3d TKG2d TKShHealing
        TKMesh TKGeomAlgo
    )
endif()

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
