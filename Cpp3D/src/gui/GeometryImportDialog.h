#pragma once
#include "geometry/STEPImporter.h"
#include "geometry/Mesh3D.h"
#include <QDialog>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>

namespace BEMCS {

// ============================================================================
// Geometry Import Dialog
//
// Allows user to:
// - Browse and select STEP files
// - Set tessellation quality
// - Set voltage for the imported geometry
// - Preview mesh statistics before applying
// ============================================================================
class GeometryImportDialog : public QDialog {
    Q_OBJECT

public:
    explicit GeometryImportDialog(QWidget* parent = nullptr);

    // Get imported surface mesh (valid after successful import)
    const SurfaceMesh& getMesh() const { return importedMesh_; }
    bool hasValidMesh() const { return !importedMesh_.empty(); }

    // Get user-specified voltage for the geometry
    double getVoltage() const;

    // Get tessellation deflection parameter
    double getDeflection() const;

    // Get scale factor (e.g., if STEP is in meters, convert to mm)
    double getScaleFactor() const;

private slots:
    void browseFile();
    void doImport();

private:
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
};

} // namespace BEMCS
