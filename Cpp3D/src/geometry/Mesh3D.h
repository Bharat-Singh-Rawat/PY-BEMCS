#pragma once
#include "core/Vec3.h"
#include <vector>
#include <array>
#include <string>

namespace BEMCS {

// ============================================================================
// 3D Mesh representation (supports both structured and unstructured)
// ============================================================================

// Triangle for surface mesh
struct Triangle {
    std::array<int, 3> vertices; // Indices into vertex array
    Vec3 normal;
    int faceId = -1; // CAD face index from STEP topology
    int bodyId = -1; // CAD solid/body index from STEP topology
};

// Tetrahedron for volume mesh
struct Tetrahedron {
    std::array<int, 4> vertices;
    double volume = 0.0;
    int materialId = 0;
};

// Surface mesh (for geometry representation and boundary conditions)
struct SurfaceMesh {
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;

    void clear() { vertices.clear(); triangles.clear(); }
    bool empty() const { return triangles.empty(); }

    // Compute axis-aligned bounding box
    void getBoundingBox(Vec3& minPt, Vec3& maxPt) const;

    // Check if a point is inside the closed surface (ray casting)
    bool isPointInside(const Vec3& pt) const;
};

// Volume mesh (for FEM-type solvers if needed)
struct VolumeMesh {
    std::vector<Vec3> vertices;
    std::vector<Tetrahedron> tets;

    void clear() { vertices.clear(); tets.clear(); }
    bool empty() const { return tets.empty(); }
};

// ============================================================================
// Mesh quality metrics
// ============================================================================
struct MeshStats {
    int numVertices = 0;
    int numTriangles = 0;
    int numTetrahedra = 0;
    double minEdgeLength = 0.0;
    double maxEdgeLength = 0.0;
    double avgEdgeLength = 0.0;
    double minAspectRatio = 0.0;
    Vec3 bboxMin, bboxMax;
};

} // namespace BEMCS
