#pragma once
#include "geometry/STEPImporter.h"
#include "geometry/Mesh3D.h"
#include <QDialog>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QTreeWidget>
#include <QMap>
#include <QSet>
#include <QCheckBox>

#ifdef USE_VTK
#include <QVTKOpenGLNativeWidget.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkActor.h>
#include <vtkPolyData.h>
#include <vtkCellPicker.h>
#include <vtkIntArray.h>
#include <vtkPlane.h>
#endif

namespace BEMCS {

// ============================================================================
// Geometry Import Dialog
//
// Allows user to:
// - Browse and select STEP files
// - Set tessellation quality
// - Set voltage for the imported geometry
// - Preview mesh in 3D and click faces/bodies to select them
// - Cut view along XZ/XY/YZ to see inner faces
// ============================================================================
class GeometryImportDialog : public QDialog {
    Q_OBJECT

public:
    explicit GeometryImportDialog(QWidget* parent = nullptr);

    const SurfaceMesh& getMesh() const { return importedMesh_; }
    bool hasValidMesh() const { return !importedMesh_.empty(); }

    double getVoltage() const;
    double getDeflection() const;
    double getScaleFactor() const;

    // Face/body assignments
    QMap<int, double> getFaceVoltages() const { return faceVoltages_; }
    int getSourceFaceId() const { return sourceFaceId_; }

private slots:
    void browseFile();
    void doImport();
    void onAssignVoltage();
    void onSetSource();
    void onTreeSelectionChanged();
    void onSelectionModeChanged();
    void onCutPlaneToggled(bool enabled);
    void onCutPlaneAxisChanged(int index);

private:
    void populateFaceTree();
    void populateBodyTree();
    void rebuildTree();
    void buildPreviewMesh();
    void highlightInPreview(const QSet<int>& faceIds);
    void updateClipPlane();
    QSet<int> getSelectedFaceIds() const;

    QLineEdit*       editFilePath_;
    QDoubleSpinBox*  spinDeflection_;
    QDoubleSpinBox*  spinVoltage_;
    QDoubleSpinBox*  spinScale_;
    QComboBox*       comboUnits_;
    QLabel*          lblStatus_;
    QPushButton*     btnImport_;
    QPushButton*     btnApply_;

    SurfaceMesh importedMesh_;
    STEPImporter importer_;

    // Selection mode
    QComboBox*   comboSelectionMode_; // Face / Body
    bool isBodyMode() const;

    // Face/body selection
    QTreeWidget* faceTree_;
    QPushButton* btnAssignVoltage_;
    QPushButton* btnSetSource_;
    QMap<int, double> faceVoltages_;
    int sourceFaceId_ = -1;

    // Mapping: bodyId -> set of faceIds
    QMap<int, QSet<int>> bodyToFaces_;

    // Cut plane controls
    QCheckBox*   chkCutPlane_;
    QComboBox*   comboCutAxis_;

    // 3D preview with picking
#ifdef USE_VTK
    QVTKOpenGLNativeWidget* previewWidget_;
    vtkSmartPointer<vtkRenderer> previewRenderer_;
    vtkSmartPointer<vtkActor> meshActor_;
    vtkSmartPointer<vtkPolyData> previewPolyData_;
    vtkSmartPointer<vtkIntArray> cellFaceIds_;   // VTK cell -> CAD faceId
    vtkSmartPointer<vtkIntArray> cellBodyIds_;   // VTK cell -> CAD bodyId
    vtkSmartPointer<vtkCellPicker> cellPicker_;
    vtkSmartPointer<vtkPlane> clipPlane_;

    void setupPickerCallback();
    void onCellPicked(vtkIdType cellId);
    void applyClipToMapper();
#endif

    bool ignoreTreeSelection_ = false;
};

} // namespace BEMCS
