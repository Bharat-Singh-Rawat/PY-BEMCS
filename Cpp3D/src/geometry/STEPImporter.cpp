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
#include <TopoDS_Solid.hxx>
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
    // Iterate by solid first to track body IDs, then faces within each solid.
    // If the shape has no solids (e.g. a shell), fall back to flat face iteration.
    int globalVertexOffset = 0;
    int cadFaceIndex = 0;

    auto extractFace = [&](const TopoDS_Face& face, int bodyIdx) {
        TopLoc_Location location;
        Handle(Poly_Triangulation) triangulation =
            BRep_Tool::Triangulation(face, location);

        if (triangulation.IsNull()) return;

        int nbNodes = triangulation->NbNodes();
        int nbTris = triangulation->NbTriangles();

        for (int i = 1; i <= nbNodes; i++) {
            gp_Pnt pt = triangulation->Node(i);
            pt.Transform(location.Transformation());
            outMesh.vertices.push_back(Vec3(pt.X(), pt.Y(), pt.Z()));
        }

        for (int i = 1; i <= nbTris; i++) {
            int n1, n2, n3;
            triangulation->Triangle(i).Get(n1, n2, n3);

            Triangle tri;
            tri.vertices[0] = globalVertexOffset + n1 - 1;
            tri.vertices[1] = globalVertexOffset + n2 - 1;
            tri.vertices[2] = globalVertexOffset + n3 - 1;

            const Vec3& v0 = outMesh.vertices[tri.vertices[0]];
            const Vec3& v1 = outMesh.vertices[tri.vertices[1]];
            const Vec3& v2 = outMesh.vertices[tri.vertices[2]];
            Vec3 e1 = v1 - v0;
            Vec3 e2 = v2 - v0;
            tri.normal = e1.cross(e2).normalized();

            if (face.Orientation() == TopAbs_REVERSED) {
                std::swap(tri.vertices[1], tri.vertices[2]);
                tri.normal = tri.normal * -1.0;
            }

            tri.faceId = cadFaceIndex;
            tri.bodyId = bodyIdx;
            outMesh.triangles.push_back(tri);
        }

        globalVertexOffset += nbNodes;
        cadFaceIndex++;
    };

    // Check if there are solids in the shape
    bool hasSolids = false;
    for (TopExp_Explorer solidExp(shape, TopAbs_SOLID); solidExp.More();
         solidExp.Next()) {
        hasSolids = true;
        break;
    }

    if (hasSolids) {
        int bodyIndex = 0;
        for (TopExp_Explorer solidExp(shape, TopAbs_SOLID); solidExp.More();
             solidExp.Next(), bodyIndex++) {
            TopoDS_Shape solid = solidExp.Current();
            for (TopExp_Explorer faceExp(solid, TopAbs_FACE); faceExp.More();
                 faceExp.Next()) {
                extractFace(TopoDS::Face(faceExp.Current()), bodyIndex);
            }
        }
    } else {
        // No solids — iterate faces directly, all body 0
        for (TopExp_Explorer faceExp(shape, TopAbs_FACE); faceExp.More();
             faceExp.Next()) {
            extractFace(TopoDS::Face(faceExp.Current()), 0);
        }
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
