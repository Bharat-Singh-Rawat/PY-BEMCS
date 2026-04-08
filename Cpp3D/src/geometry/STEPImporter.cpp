#include "geometry/STEPImporter.h"
#include <algorithm>
#include <cctype>

#ifdef USE_OCCT
#include <STEPControl_Reader.hxx>
#include <BRep_Tool.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <Poly_Triangulation.hxx>
#include <TopLoc_Location.hxx>
#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>
#include <Standard_Handle.hxx>
#endif

namespace BEMCS {

bool STEPImporter::isSupported(const std::string& filePath) {
    std::string ext = filePath.substr(filePath.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "step" || ext == "stp");
}

bool STEPImporter::import(const std::string& filePath, SurfaceMesh& outMesh,
                           double meshDeflection) {
    outMesh.clear();

#ifdef USE_OCCT
    // ── Read STEP file ─────────────────────────────────────────────────
    STEPControl_Reader reader;
    IFSelect_ReturnStatus status = reader.ReadFile(filePath.c_str());

    if (status != IFSelect_RetDone) {
        lastError_ = "Failed to read STEP file: " + filePath;
        return false;
    }

    reader.TransferRoots();
    TopoDS_Shape shape = reader.OneShape();

    if (shape.IsNull()) {
        lastError_ = "No valid shape found in STEP file";
        return false;
    }

    // ── Tessellate the shape ───────────────────────────────────────────
    BRepMesh_IncrementalMesh mesher(shape, meshDeflection, Standard_False,
                                    meshDeflection * 0.5, Standard_True);
    mesher.Perform();

    if (!mesher.IsDone()) {
        lastError_ = "Tessellation failed";
        return false;
    }

    // ── Extract triangulated faces ─────────────────────────────────────
    int globalVertexOffset = 0;

    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More();
         explorer.Next()) {
        TopoDS_Face face = TopoDS::Face(explorer.Current());
        TopLoc_Location location;

        Handle(Poly_Triangulation) triangulation =
            BRep_Tool::Triangulation(face, location);

        if (triangulation.IsNull()) continue;

        int nbNodes = triangulation->NbNodes();
        int nbTris = triangulation->NbTriangles();

        // Extract vertices
        for (int i = 1; i <= nbNodes; i++) {
            gp_Pnt pt = triangulation->Node(i);
            pt.Transform(location.Transformation());
            // STEP files typically use mm, keep as-is
            outMesh.vertices.push_back(Vec3(pt.X(), pt.Y(), pt.Z()));
        }

        // Extract triangles
        for (int i = 1; i <= nbTris; i++) {
            int n1, n2, n3;
            triangulation->Triangle(i).Get(n1, n2, n3);

            // Convert from 1-based OCCT indexing to 0-based
            Triangle tri;
            tri.vertices[0] = globalVertexOffset + n1 - 1;
            tri.vertices[1] = globalVertexOffset + n2 - 1;
            tri.vertices[2] = globalVertexOffset + n3 - 1;

            // Compute normal
            const Vec3& v0 = outMesh.vertices[tri.vertices[0]];
            const Vec3& v1 = outMesh.vertices[tri.vertices[1]];
            const Vec3& v2 = outMesh.vertices[tri.vertices[2]];
            Vec3 e1 = v1 - v0;
            Vec3 e2 = v2 - v0;
            tri.normal = e1.cross(e2).normalized();

            // Handle face orientation
            if (face.Orientation() == TopAbs_REVERSED) {
                std::swap(tri.vertices[1], tri.vertices[2]);
                tri.normal = tri.normal * -1.0;
            }

            outMesh.triangles.push_back(tri);
        }

        globalVertexOffset += nbNodes;
    }

    if (outMesh.triangles.empty()) {
        lastError_ = "No triangles extracted from STEP file";
        return false;
    }

    return true;

#else
    lastError_ = "OpenCASCADE not available. Rebuild with -DUSE_OCCT=ON";
    return false;
#endif
}

} // namespace BEMCS
