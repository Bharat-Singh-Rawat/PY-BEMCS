#pragma once
#include "Mesh3D.h"
#include <string>

namespace BEMCS {

// ============================================================================
// STEP File Importer using OpenCASCADE Technology (OCCT)
//
// Reads CAD geometry from STEP files and converts to triangulated
// surface mesh for use in the PIC simulation.
// ============================================================================
class STEPImporter {
public:
    // Import a STEP file and return triangulated surface mesh
    // Units are converted to mm.
    // meshDeflection controls tessellation quality (smaller = finer)
    bool import(const std::string& filePath, SurfaceMesh& outMesh,
                double meshDeflection = 0.1);

    // Get last error message
    const std::string& getError() const { return lastError_; }

    // Supported file extensions
    static bool isSupported(const std::string& filePath);

private:
    std::string lastError_;
};

} // namespace BEMCS
