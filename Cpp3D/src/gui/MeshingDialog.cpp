#include "gui/MeshingDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QDialogButtonBox>

namespace BEMCS {

MeshingDialog::MeshingDialog(const MeshGenerator::MeshSettings& current,
                              QWidget* parent)
    : QDialog(parent) {
    setWindowTitle("3D Meshing Options");
    setMinimumWidth(400);

    auto mainLay = new QVBoxLayout(this);

    // ── Mesh parameters group ──────────────────────────────────────────
    auto paramGroup = new QGroupBox("Mesh Parameters");
    auto formLay = new QFormLayout();

    spinCellSize_ = new QDoubleSpinBox();
    spinCellSize_->setRange(0.005, 5.0);
    spinCellSize_->setValue(current.cellSize_mm);
    spinCellSize_->setSingleStep(0.01);
    spinCellSize_->setDecimals(3);
    spinCellSize_->setSuffix(" mm");
    formLay->addRow("Cell Size:", spinCellSize_);

    spinRefine_ = new QDoubleSpinBox();
    spinRefine_->setRange(0.1, 10.0);
    spinRefine_->setValue(current.refineFactor);
    spinRefine_->setSingleStep(0.5);
    spinRefine_->setDecimals(1);
    formLay->addRow("Refinement Factor:", spinRefine_);

    spinMaxCells_ = new QSpinBox();
    spinMaxCells_->setRange(10000, 100000000);
    spinMaxCells_->setValue(current.maxCells);
    spinMaxCells_->setSingleStep(1000000);
    formLay->addRow("Max Cells:", spinMaxCells_);

    chkAutoSize_ = new QCheckBox("Auto-compute cell size from geometry");
    chkAutoSize_->setChecked(current.autoSize);
    formLay->addRow(chkAutoSize_);

    spinPadding_ = new QDoubleSpinBox();
    spinPadding_->setRange(0.0, 50.0);
    spinPadding_->setValue(current.padding_mm);
    spinPadding_->setSingleStep(0.5);
    spinPadding_->setDecimals(1);
    spinPadding_->setSuffix(" mm");
    formLay->addRow("Domain Padding:", spinPadding_);

    paramGroup->setLayout(formLay);
    mainLay->addWidget(paramGroup);

    // ── Statistics display ─────────────────────────────────────────────
    auto statsGroup = new QGroupBox("Mesh Statistics");
    auto statsLay = new QVBoxLayout();
    lblStats_ = new QLabel("No mesh generated yet.");
    lblStats_->setWordWrap(true);
    statsLay->addWidget(lblStats_);
    statsGroup->setLayout(statsLay);
    mainLay->addWidget(statsGroup);

    // ── Buttons ────────────────────────────────────────────────────────
    auto buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    mainLay->addWidget(buttons);
}

MeshGenerator::MeshSettings MeshingDialog::getSettings() const {
    MeshGenerator::MeshSettings s;
    s.cellSize_mm   = spinCellSize_->value();
    s.refineFactor  = spinRefine_->value();
    s.maxCells      = spinMaxCells_->value();
    s.autoSize      = chkAutoSize_->isChecked();
    s.padding_mm    = spinPadding_->value();
    return s;
}

void MeshingDialog::showStats(const MeshStats& stats) {
    QString text;
    text += QString("Vertices: %1\n").arg(stats.numVertices);
    text += QString("Boundary Cells: %1\n").arg(stats.numTriangles);
    text += QString("Edge Lengths: min=%1, max=%2, avg=%3 mm\n")
                .arg(stats.minEdgeLength, 0, 'f', 4)
                .arg(stats.maxEdgeLength, 0, 'f', 4)
                .arg(stats.avgEdgeLength, 0, 'f', 4);
    text += QString("Bounding Box: (%1, %2, %3) to (%4, %5, %6) mm")
                .arg(stats.bboxMin.x, 0, 'f', 2)
                .arg(stats.bboxMin.y, 0, 'f', 2)
                .arg(stats.bboxMin.z, 0, 'f', 2)
                .arg(stats.bboxMax.x, 0, 'f', 2)
                .arg(stats.bboxMax.y, 0, 'f', 2)
                .arg(stats.bboxMax.z, 0, 'f', 2);
    lblStats_->setText(text);
}

} // namespace BEMCS
