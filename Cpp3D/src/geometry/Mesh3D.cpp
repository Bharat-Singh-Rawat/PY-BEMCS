#include "geometry/Mesh3D.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace BEMCS {

void SurfaceMesh::getBoundingBox(Vec3& minPt, Vec3& maxPt) const {
    if (vertices.empty()) {
        minPt = maxPt = Vec3(0, 0, 0);
        return;
    }

    minPt = {std::numeric_limits<double>::max(),
             std::numeric_limits<double>::max(),
             std::numeric_limits<double>::max()};
    maxPt = {std::numeric_limits<double>::lowest(),
             std::numeric_limits<double>::lowest(),
             std::numeric_limits<double>::lowest()};

    for (const auto& v : vertices) {
        minPt.x = std::min(minPt.x, v.x);
        minPt.y = std::min(minPt.y, v.y);
        minPt.z = std::min(minPt.z, v.z);
        maxPt.x = std::max(maxPt.x, v.x);
        maxPt.y = std::max(maxPt.y, v.y);
        maxPt.z = std::max(maxPt.z, v.z);
    }
}

bool SurfaceMesh::isPointInside(const Vec3& pt) const {
    // Ray casting algorithm along +X direction
    int crossings = 0;

    for (const auto& tri : triangles) {
        const Vec3& v0 = vertices[tri.vertices[0]];
        const Vec3& v1 = vertices[tri.vertices[1]];
        const Vec3& v2 = vertices[tri.vertices[2]];

        // Check if ray from pt in +X direction intersects triangle
        Vec3 e1 = v1 - v0;
        Vec3 e2 = v2 - v0;
        Vec3 dir(1.0, 0.0, 0.0); // Ray direction

        Vec3 h = dir.cross(e2);
        double a = e1.dot(h);
        if (std::abs(a) < 1e-12) continue;

        double f = 1.0 / a;
        Vec3 s = pt - v0;
        double u = f * s.dot(h);
        if (u < 0.0 || u > 1.0) continue;

        Vec3 q = s.cross(e1);
        double v = f * dir.dot(q);
        if (v < 0.0 || u + v > 1.0) continue;

        double t = f * e2.dot(q);
        if (t > 1e-8) crossings++;
    }

    return (crossings % 2) == 1;
}

} // namespace BEMCS
