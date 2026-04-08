#pragma once
#include "geometry/MeshGenerator.h"
#include <QDialog>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>

namespace BEMCS {

// ============================================================================
// Meshing Options Dialog
//
// Allows user to configure mesh generation parameters:
// - Cell size
// - Refinement factor
// - Max cell count
// - Auto-sizing
// - Domain padding
// ============================================================================
class MeshingDialog : public QDialog {
    Q_OBJECT

public:
    explicit MeshingDialog(const MeshGenerator::MeshSettings& current,
                           QWidget* parent = nullptr);

    MeshGenerator::MeshSettings getSettings() const;

    // Show mesh statistics after generation
    void showStats(const MeshStats& stats);

private:
    QDoubleSpinBox* spinCellSize_;
    QDoubleSpinBox* spinRefine_;
    QSpinBox*       spinMaxCells_;
    QCheckBox*      chkAutoSize_;
    QDoubleSpinBox* spinPadding_;
    QLabel*         lblStats_;
};

} // namespace BEMCS
