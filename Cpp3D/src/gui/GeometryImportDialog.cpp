#include "gui/GeometryImportDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QInputDialog>
#include <QHeaderView>

namespace BEMCS {

GeometryImportDialog::GeometryImportDialog(QWidget* parent)
    : QDialog(parent) {
    setWindowTitle("Import STEP Geometry");
    setMinimumWidth(500);

    auto mainLay = new QVBoxLayout(this);

    // ── File selection ─────────────────────────────────────────────────
    auto fileGroup = new QGroupBox("STEP File");
    auto fileLay = new QHBoxLayout();
    editFilePath_ = new QLineEdit();
    editFilePath_->setPlaceholderText("Select a .step/.stp/.STEP/.STP file...");
    auto btnBrowse = new QPushButton("Browse...");
    connect(btnBrowse, &QPushButton::clicked, this, &GeometryImportDialog::browseFile);
    fileLay->addWidget(editFilePath_);
    fileLay->addWidget(btnBrowse);
    fileGroup->setLayout(fileLay);
    mainLay->addWidget(fileGroup);

    // ── Import settings ────────────────────────────────────────────────
    auto settingsGroup = new QGroupBox("Import Settings");
    auto formLay = new QFormLayout();

    spinDeflection_ = new QDoubleSpinBox();
    spinDeflection_->setRange(0.001, 10.0);
    spinDeflection_->setValue(0.1);
    spinDeflection_->setSingleStep(0.01);
    spinDeflection_->setDecimals(3);
    spinDeflection_->setToolTip("Tessellation quality. Smaller = finer mesh.");
    formLay->addRow("Tessellation Deflection:", spinDeflection_);

    comboUnits_ = new QComboBox();
    comboUnits_->addItems({"Millimeters (mm)", "Meters (m)", "Inches (in)"});
    connect(comboUnits_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [this](int idx) {
                double scales[] = {1.0, 1000.0, 25.4};
                spinScale_->setValue(scales[idx]);
            });
    formLay->addRow("Input Units:", comboUnits_);

    spinScale_ = new QDoubleSpinBox();
    spinScale_->setRange(0.001, 100000.0);
    spinScale_->setValue(1.0);
    spinScale_->setSingleStep(1.0);
    spinScale_->setDecimals(3);
    spinScale_->setToolTip("Scale factor to convert to mm");
    formLay->addRow("Scale Factor (→ mm):", spinScale_);

    spinVoltage_ = new QDoubleSpinBox();
    spinVoltage_->setRange(-10000, 50000);
    spinVoltage_->setValue(0);
    spinVoltage_->setSingleStep(100);
    spinVoltage_->setDecimals(0);
    spinVoltage_->setSuffix(" V");
    formLay->addRow("Boundary Voltage:", spinVoltage_);

    settingsGroup->setLayout(formLay);
    mainLay->addWidget(settingsGroup);

    // ── Import button & status ─────────────────────────────────────────
    btnImport_ = new QPushButton("Import && Preview");
    btnImport_->setStyleSheet("QPushButton { background-color: #2196F3; color: white; "
                               "font-weight: bold; padding: 8px; }");
    connect(btnImport_, &QPushButton::clicked, this, &GeometryImportDialog::doImport);
    mainLay->addWidget(btnImport_);

    lblStatus_ = new QLabel("No file imported.");
    lblStatus_->setWordWrap(true);
    mainLay->addWidget(lblStatus_);

    // ── Face / Body Assignment ────────────────────────────────────────
    auto faceGroup = new QGroupBox("Face / Body Assignment");
    auto faceLay = new QVBoxLayout();

    faceTree_ = new QTreeWidget();
    faceTree_->setHeaderLabels({"Face", "Type", "Voltage (V)"});
    faceTree_->header()->setStretchLastSection(true);
    faceTree_->setMinimumHeight(120);
    faceTree_->setSelectionMode(QAbstractItemView::SingleSelection);
    faceLay->addWidget(faceTree_);

    auto faceBtnRow = new QHBoxLayout();
    btnAssignVoltage_ = new QPushButton("Set Voltage");
    btnAssignVoltage_->setEnabled(false);
    btnAssignVoltage_->setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 5px; }");
    connect(btnAssignVoltage_, &QPushButton::clicked, this, &GeometryImportDialog::onAssignVoltage);
    faceBtnRow->addWidget(btnAssignVoltage_);

    btnSetSource_ = new QPushButton("Set as Particle Source");
    btnSetSource_->setEnabled(false);
    btnSetSource_->setStyleSheet("QPushButton { background-color: #9C27B0; color: white; padding: 5px; }");
    connect(btnSetSource_, &QPushButton::clicked, this, &GeometryImportDialog::onSetSource);
    faceBtnRow->addWidget(btnSetSource_);

    faceLay->addLayout(faceBtnRow);
    faceGroup->setLayout(faceLay);
    mainLay->addWidget(faceGroup);

    // ── Dialog buttons ─────────────────────────────────────────────────
    btnApply_ = new QPushButton("Apply to Simulation");
    btnApply_->setEnabled(false);
    btnApply_->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; "
                              "font-weight: bold; padding: 8px; }");
    connect(btnApply_, &QPushButton::clicked, this, &QDialog::accept);

    auto btnCancel = new QPushButton("Cancel");
    connect(btnCancel, &QPushButton::clicked, this, &QDialog::reject);

    auto btnRow = new QHBoxLayout();
    btnRow->addWidget(btnApply_);
    btnRow->addWidget(btnCancel);
    mainLay->addLayout(btnRow);
}

void GeometryImportDialog::browseFile() {
    QString path = QFileDialog::getOpenFileName(
        this, "Select STEP File", QString(),
        "STEP Files (*.step *.stp *.STEP *.STP *.Step);;All Files (*)");

    if (!path.isEmpty()) {
        editFilePath_->setText(path);
    }
}

void GeometryImportDialog::doImport() {
    QString path = editFilePath_->text();
    if (path.isEmpty()) {
        QMessageBox::warning(this, "No File", "Please select a STEP file first.");
        return;
    }

    lblStatus_->setText("Importing...");
    QApplication::processEvents();

    bool ok = importer_.import(path.toStdString(), importedMesh_,
                                spinDeflection_->value());

    if (!ok) {
        lblStatus_->setText("Import failed: " +
                            QString::fromStdString(importer_.getError()));
        btnApply_->setEnabled(false);
        return;
    }

    // Apply scale factor
    double scale = spinScale_->value();
    if (std::abs(scale - 1.0) > 1e-6) {
        for (auto& v : importedMesh_.vertices) {
            v.x *= scale;
            v.y *= scale;
            v.z *= scale;
        }
    }

    // Show statistics
    Vec3 minPt, maxPt;
    importedMesh_.getBoundingBox(minPt, maxPt);

    QString status;
    status += QString("Import successful!\n");
    status += QString("Vertices: %1\n").arg(importedMesh_.vertices.size());
    status += QString("Triangles: %1\n").arg(importedMesh_.triangles.size());
    status += QString("Bounding Box: (%1, %2, %3) to (%4, %5, %6) mm")
                  .arg(minPt.x, 0, 'f', 2).arg(minPt.y, 0, 'f', 2)
                  .arg(minPt.z, 0, 'f', 2)
                  .arg(maxPt.x, 0, 'f', 2).arg(maxPt.y, 0, 'f', 2)
                  .arg(maxPt.z, 0, 'f', 2);
    lblStatus_->setText(status);
    btnApply_->setEnabled(true);

    // Populate the face tree
    populateFaceTree();
    btnAssignVoltage_->setEnabled(true);
    btnSetSource_->setEnabled(true);
}

double GeometryImportDialog::getVoltage() const {
    return spinVoltage_->value();
}

double GeometryImportDialog::getDeflection() const {
    return spinDeflection_->value();
}

double GeometryImportDialog::getScaleFactor() const {
    return spinScale_->value();
}

void GeometryImportDialog::populateFaceTree() {
    faceTree_->clear();
    faceVoltages_.clear();
    sourceFaceId_ = -1;

    if (importedMesh_.empty()) return;

    // Group triangles into logical faces (~50 triangles each, max 20 faces)
    int totalTri = static_cast<int>(importedMesh_.triangles.size());
    int triPerFace = std::max(1, totalTri / 20);
    int numFaces = std::max(1, (totalTri + triPerFace - 1) / triPerFace);
    numFaces = std::min(numFaces, 20);

    for (int f = 0; f < numFaces; f++) {
        int startTri = f * triPerFace;
        int endTri = std::min(startTri + triPerFace, totalTri);
        auto item = new QTreeWidgetItem(faceTree_);
        item->setText(0, QString("Face %1 (%2 tris)").arg(f).arg(endTri - startTri));
        item->setText(1, "None");
        item->setText(2, "--");
        item->setData(0, Qt::UserRole, f);
    }

    faceTree_->resizeColumnToContents(0);
}

void GeometryImportDialog::onAssignVoltage() {
    auto items = faceTree_->selectedItems();
    if (items.isEmpty()) {
        QMessageBox::information(this, "No Selection", "Select a face first.");
        return;
    }

    auto item = items.first();
    int faceId = item->data(0, Qt::UserRole).toInt();

    bool ok;
    double voltage = QInputDialog::getDouble(this, "Set Face Voltage",
        QString("Voltage for %1 (V):").arg(item->text(0)),
        faceVoltages_.value(faceId, spinVoltage_->value()),
        -50000, 50000, 0, &ok);

    if (ok) {
        faceVoltages_[faceId] = voltage;
        item->setText(1, "Boundary");
        item->setText(2, QString::number(voltage, 'f', 0) + " V");
        item->setForeground(1, QBrush(QColor("#FF9800")));
    }
}

void GeometryImportDialog::onSetSource() {
    auto items = faceTree_->selectedItems();
    if (items.isEmpty()) {
        QMessageBox::information(this, "No Selection", "Select a face first.");
        return;
    }

    // Clear previous source marking
    for (int i = 0; i < faceTree_->topLevelItemCount(); i++) {
        auto it = faceTree_->topLevelItem(i);
        if (it->text(1) == "Source") {
            it->setText(1, "None");
            it->setForeground(1, QBrush(QColor("#e0e0e0")));
        }
    }

    auto item = items.first();
    sourceFaceId_ = item->data(0, Qt::UserRole).toInt();
    item->setText(1, "Source");
    item->setForeground(1, QBrush(QColor("#9C27B0")));
}

} // namespace BEMCS
