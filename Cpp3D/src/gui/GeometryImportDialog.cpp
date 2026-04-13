#include "gui/GeometryImportDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QInputDialog>
#include <QHeaderView>
#include <QSplitter>

#ifdef USE_VTK
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkCellArray.h>
#include <vtkTriangle.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkCellData.h>
#include <vtkCamera.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkCallbackCommand.h>
#include <vtkClipPolyData.h>
#endif

namespace BEMCS {

// ── HSV color helper (reused in build & highlight) ────────────────────
static void hsvColor(int id, unsigned char rgb[3]) {
    double hue = std::fmod(id * 0.618033988749895, 1.0);
    double s = 0.55, v = 0.85;
    int hi = static_cast<int>(hue * 6.0) % 6;
    double f = hue * 6.0 - hi;
    double p = v * (1.0 - s);
    double q = v * (1.0 - f * s);
    double t = v * (1.0 - (1.0 - f) * s);
    double r, g, b;
    switch (hi) {
        case 0: r=v; g=t; b=p; break;
        case 1: r=q; g=v; b=p; break;
        case 2: r=p; g=v; b=t; break;
        case 3: r=p; g=q; b=v; break;
        case 4: r=t; g=p; b=v; break;
        default:r=v; g=p; b=q; break;
    }
    rgb[0] = static_cast<unsigned char>(r * 255);
    rgb[1] = static_cast<unsigned char>(g * 255);
    rgb[2] = static_cast<unsigned char>(b * 255);
}

GeometryImportDialog::GeometryImportDialog(QWidget* parent)
    : QDialog(parent) {
    setWindowTitle("Import STEP Geometry");
    setMinimumSize(1100, 750);

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

    // ── Splitter: 3D Preview (left) | Face/Body Tree (right) ──────────
    auto splitter = new QSplitter(Qt::Horizontal);

#ifdef USE_VTK
    // 3D Preview widget with cut plane controls
    auto previewGroup = new QGroupBox("3D Preview (click to select)");
    auto previewLay = new QVBoxLayout();

    // Cut plane toolbar
    auto cutRow = new QHBoxLayout();
    chkCutPlane_ = new QCheckBox("Cut Plane");
    chkCutPlane_->setToolTip("Clip geometry in half to reveal inner faces");
    connect(chkCutPlane_, &QCheckBox::toggled,
            this, &GeometryImportDialog::onCutPlaneToggled);
    cutRow->addWidget(chkCutPlane_);

    comboCutAxis_ = new QComboBox();
    comboCutAxis_->addItems({"XZ (cut along Y)", "XY (cut along Z)", "YZ (cut along X)"});
    comboCutAxis_->setEnabled(false);
    connect(comboCutAxis_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GeometryImportDialog::onCutPlaneAxisChanged);
    cutRow->addWidget(comboCutAxis_);
    cutRow->addStretch();
    previewLay->addLayout(cutRow);

    previewWidget_ = new QVTKOpenGLNativeWidget();
    previewWidget_->setMinimumSize(400, 300);

    auto renWin = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    previewWidget_->setRenderWindow(renWin);

    previewRenderer_ = vtkSmartPointer<vtkRenderer>::New();
    previewRenderer_->SetBackground(0.12, 0.12, 0.18);
    previewRenderer_->SetBackground2(0.22, 0.22, 0.32);
    previewRenderer_->GradientBackgroundOn();
    renWin->AddRenderer(previewRenderer_);

    meshActor_ = vtkSmartPointer<vtkActor>::New();
    previewRenderer_->AddActor(meshActor_);

    cellPicker_ = vtkSmartPointer<vtkCellPicker>::New();
    cellPicker_->SetTolerance(0.005);

    clipPlane_ = vtkSmartPointer<vtkPlane>::New();

    previewLay->addWidget(previewWidget_);
    previewGroup->setLayout(previewLay);
    splitter->addWidget(previewGroup);

    setupPickerCallback();
#endif

    // ── Face / Body Assignment ────────────────────────────────────────
    auto faceGroup = new QGroupBox("Face / Body Assignment");
    auto faceLay = new QVBoxLayout();

    // Selection mode toggle
    auto modeRow = new QHBoxLayout();
    auto modeLabel = new QLabel("Selection Mode:");
    comboSelectionMode_ = new QComboBox();
    comboSelectionMode_->addItems({"Face", "Body"});
    comboSelectionMode_->setToolTip("Face: select individual CAD faces\n"
                                    "Body: select entire solids/bodies at once");
    connect(comboSelectionMode_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GeometryImportDialog::onSelectionModeChanged);
    modeRow->addWidget(modeLabel);
    modeRow->addWidget(comboSelectionMode_);
    modeRow->addStretch();
    faceLay->addLayout(modeRow);

    faceTree_ = new QTreeWidget();
    faceTree_->setHeaderLabels({"Face", "Type", "Voltage (V)"});
    faceTree_->header()->setStretchLastSection(true);
    faceTree_->setMinimumHeight(120);
    faceTree_->setSelectionMode(QAbstractItemView::SingleSelection);
    connect(faceTree_, &QTreeWidget::itemSelectionChanged,
            this, &GeometryImportDialog::onTreeSelectionChanged);
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
    splitter->addWidget(faceGroup);

    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 2);
    mainLay->addWidget(splitter, 1);

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

bool GeometryImportDialog::isBodyMode() const {
    return comboSelectionMode_->currentIndex() == 1;
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

    // Build body-to-face mapping
    bodyToFaces_.clear();
    for (const auto& tri : importedMesh_.triangles) {
        bodyToFaces_[tri.bodyId].insert(tri.faceId);
    }

    // Show statistics
    Vec3 minPt, maxPt;
    importedMesh_.getBoundingBox(minPt, maxPt);

    // Count unique faces and bodies
    QSet<int> uniqueFaces, uniqueBodies;
    for (const auto& tri : importedMesh_.triangles) {
        uniqueFaces.insert(tri.faceId);
        uniqueBodies.insert(tri.bodyId);
    }

    QString status;
    status += QString("Import successful!\n");
    status += QString("Vertices: %1  |  Triangles: %2\n")
                  .arg(importedMesh_.vertices.size())
                  .arg(importedMesh_.triangles.size());
    status += QString("CAD Faces: %1  |  Bodies: %2\n")
                  .arg(uniqueFaces.size()).arg(uniqueBodies.size());
    status += QString("Bounding Box: (%1, %2, %3) to (%4, %5, %6) mm")
                  .arg(minPt.x, 0, 'f', 2).arg(minPt.y, 0, 'f', 2)
                  .arg(minPt.z, 0, 'f', 2)
                  .arg(maxPt.x, 0, 'f', 2).arg(maxPt.y, 0, 'f', 2)
                  .arg(maxPt.z, 0, 'f', 2);
    lblStatus_->setText(status);
    btnApply_->setEnabled(true);

    rebuildTree();
    btnAssignVoltage_->setEnabled(true);
    btnSetSource_->setEnabled(true);

    buildPreviewMesh();
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

// ── Tree population ───────────────────────────────────────────────────

void GeometryImportDialog::rebuildTree() {
    if (isBodyMode()) {
        populateBodyTree();
    } else {
        populateFaceTree();
    }
}

void GeometryImportDialog::populateFaceTree() {
    faceTree_->clear();
    if (importedMesh_.empty()) return;

    faceTree_->setHeaderLabels({"Face", "Type", "Voltage (V)"});

    QMap<int, int> faceTriCount;
    for (const auto& tri : importedMesh_.triangles) {
        faceTriCount[tri.faceId]++;
    }

    QList<int> faceIds = faceTriCount.keys();
    std::sort(faceIds.begin(), faceIds.end());

    for (int faceId : faceIds) {
        auto item = new QTreeWidgetItem(faceTree_);
        item->setText(0, QString("Face %1 (%2 tris)").arg(faceId).arg(faceTriCount[faceId]));

        // Restore any existing assignment
        if (faceVoltages_.contains(faceId)) {
            item->setText(1, "Boundary");
            item->setText(2, QString::number(faceVoltages_[faceId], 'f', 0) + " V");
            item->setForeground(1, QBrush(QColor("#FF9800")));
        } else if (faceId == sourceFaceId_) {
            item->setText(1, "Source");
            item->setForeground(1, QBrush(QColor("#9C27B0")));
            item->setText(2, "--");
        } else {
            item->setText(1, "None");
            item->setText(2, "--");
        }

        item->setData(0, Qt::UserRole, faceId);       // face id
        item->setData(0, Qt::UserRole + 1, -1);       // body id (N/A in face mode)
    }

    faceTree_->resizeColumnToContents(0);
}

void GeometryImportDialog::populateBodyTree() {
    faceTree_->clear();
    if (importedMesh_.empty()) return;

    faceTree_->setHeaderLabels({"Body", "Type", "Voltage (V)"});

    // Count triangles per body
    QMap<int, int> bodyTriCount;
    for (const auto& tri : importedMesh_.triangles) {
        bodyTriCount[tri.bodyId]++;
    }

    QList<int> bodyIds = bodyTriCount.keys();
    std::sort(bodyIds.begin(), bodyIds.end());

    for (int bodyId : bodyIds) {
        int numFaces = bodyToFaces_[bodyId].size();
        auto item = new QTreeWidgetItem(faceTree_);
        item->setText(0, QString("Body %1 (%2 faces, %3 tris)")
                          .arg(bodyId).arg(numFaces).arg(bodyTriCount[bodyId]));

        // Check if all faces of this body have same voltage
        bool allSameVoltage = true;
        bool anyVoltage = false;
        double firstV = 0;
        bool isSource = false;
        for (int fid : bodyToFaces_[bodyId]) {
            if (faceVoltages_.contains(fid)) {
                if (!anyVoltage) { firstV = faceVoltages_[fid]; anyVoltage = true; }
                else if (std::abs(faceVoltages_[fid] - firstV) > 0.1) { allSameVoltage = false; }
            }
            if (fid == sourceFaceId_) isSource = true;
        }

        if (isSource) {
            item->setText(1, "Source");
            item->setForeground(1, QBrush(QColor("#9C27B0")));
            item->setText(2, "--");
        } else if (anyVoltage && allSameVoltage) {
            item->setText(1, "Boundary");
            item->setText(2, QString::number(firstV, 'f', 0) + " V");
            item->setForeground(1, QBrush(QColor("#FF9800")));
        } else if (anyVoltage) {
            item->setText(1, "Mixed");
            item->setText(2, "...");
            item->setForeground(1, QBrush(QColor("#FF9800")));
        } else {
            item->setText(1, "None");
            item->setText(2, "--");
        }

        item->setData(0, Qt::UserRole, -1);            // face id (N/A in body mode)
        item->setData(0, Qt::UserRole + 1, bodyId);    // body id
    }

    faceTree_->resizeColumnToContents(0);
}

QSet<int> GeometryImportDialog::getSelectedFaceIds() const {
    QSet<int> result;
    auto items = faceTree_->selectedItems();
    if (items.isEmpty()) return result;

    auto item = items.first();
    if (isBodyMode()) {
        int bodyId = item->data(0, Qt::UserRole + 1).toInt();
        if (bodyToFaces_.contains(bodyId))
            result = bodyToFaces_[bodyId];
    } else {
        int faceId = item->data(0, Qt::UserRole).toInt();
        result.insert(faceId);
    }
    return result;
}

void GeometryImportDialog::onSelectionModeChanged() {
    rebuildTree();
    highlightInPreview({}); // clear highlight
}

// ── Cut plane ─────────────────────────────────────────────────────────

void GeometryImportDialog::onCutPlaneToggled(bool enabled) {
    comboCutAxis_->setEnabled(enabled);
    applyClipToMapper();
}

void GeometryImportDialog::onCutPlaneAxisChanged(int /*index*/) {
    applyClipToMapper();
}

void GeometryImportDialog::updateClipPlane() {
#ifdef USE_VTK
    if (!previewPolyData_ || importedMesh_.empty()) return;

    Vec3 minPt, maxPt;
    importedMesh_.getBoundingBox(minPt, maxPt);
    double cx = (minPt.x + maxPt.x) * 0.5;
    double cy = (minPt.y + maxPt.y) * 0.5;
    double cz = (minPt.z + maxPt.z) * 0.5;

    int axis = comboCutAxis_->currentIndex();
    switch (axis) {
        case 0: // XZ — cut along Y at midpoint
            clipPlane_->SetOrigin(0, cy, 0);
            clipPlane_->SetNormal(0, -1, 0);
            break;
        case 1: // XY — cut along Z at midpoint
            clipPlane_->SetOrigin(0, 0, cz);
            clipPlane_->SetNormal(0, 0, -1);
            break;
        case 2: // YZ — cut along X at midpoint
            clipPlane_->SetOrigin(cx, 0, 0);
            clipPlane_->SetNormal(-1, 0, 0);
            break;
    }
#endif
}

#ifdef USE_VTK
void GeometryImportDialog::applyClipToMapper() {
    if (!previewPolyData_) return;

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();

    if (chkCutPlane_->isChecked()) {
        updateClipPlane();
        auto clipper = vtkSmartPointer<vtkClipPolyData>::New();
        clipper->SetInputData(previewPolyData_);
        clipper->SetClipFunction(clipPlane_);
        clipper->Update();
        mapper->SetInputConnection(clipper->GetOutputPort());
    } else {
        mapper->SetInputData(previewPolyData_);
    }

    mapper->SetScalarModeToUseCellData();
    mapper->SelectColorArray("FaceColors");

    meshActor_->SetMapper(mapper);
    meshActor_->GetProperty()->SetEdgeVisibility(false);
    meshActor_->GetProperty()->SetAmbient(0.3);
    meshActor_->GetProperty()->SetDiffuse(0.7);
    meshActor_->GetProperty()->SetSpecular(0.2);

    previewWidget_->renderWindow()->Render();
}
#else
void GeometryImportDialog::applyClipToMapper() {}
#endif

// ── 3D Preview mesh ───────────────────────────────────────────────────

void GeometryImportDialog::buildPreviewMesh() {
#ifdef USE_VTK
    if (importedMesh_.empty()) return;

    auto points = vtkSmartPointer<vtkPoints>::New();
    auto cells = vtkSmartPointer<vtkCellArray>::New();
    auto colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetName("FaceColors");

    cellFaceIds_ = vtkSmartPointer<vtkIntArray>::New();
    cellFaceIds_->SetName("CadFaceId");
    cellFaceIds_->SetNumberOfComponents(1);

    cellBodyIds_ = vtkSmartPointer<vtkIntArray>::New();
    cellBodyIds_->SetName("CadBodyId");
    cellBodyIds_->SetNumberOfComponents(1);

    for (const auto& v : importedMesh_.vertices) {
        points->InsertNextPoint(v.x, v.y, v.z);
    }

    for (const auto& tri : importedMesh_.triangles) {
        vtkSmartPointer<vtkTriangle> vtri = vtkSmartPointer<vtkTriangle>::New();
        vtri->GetPointIds()->SetId(0, tri.vertices[0]);
        vtri->GetPointIds()->SetId(1, tri.vertices[1]);
        vtri->GetPointIds()->SetId(2, tri.vertices[2]);
        cells->InsertNextCell(vtri);

        unsigned char rgb[3];
        hsvColor(tri.faceId, rgb);
        colors->InsertNextTypedTuple(rgb);

        cellFaceIds_->InsertNextValue(tri.faceId);
        cellBodyIds_->InsertNextValue(tri.bodyId);
    }

    previewPolyData_ = vtkSmartPointer<vtkPolyData>::New();
    previewPolyData_->SetPoints(points);
    previewPolyData_->SetPolys(cells);
    previewPolyData_->GetCellData()->SetScalars(colors);
    previewPolyData_->GetCellData()->AddArray(cellFaceIds_);
    previewPolyData_->GetCellData()->AddArray(cellBodyIds_);

    applyClipToMapper();

    previewRenderer_->ResetCamera();
    previewWidget_->renderWindow()->Render();
#endif
}

// ── Highlight ─────────────────────────────────────────────────────────

void GeometryImportDialog::highlightInPreview(const QSet<int>& faceIds) {
#ifdef USE_VTK
    if (!previewPolyData_ || !cellFaceIds_) return;

    auto colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetName("FaceColors");

    vtkIdType numCells = cellFaceIds_->GetNumberOfTuples();
    bool hasSelection = !faceIds.isEmpty();
    unsigned char highlight[3] = {255, 255, 80};

    for (vtkIdType i = 0; i < numCells; i++) {
        int cFaceId = cellFaceIds_->GetValue(i);
        if (hasSelection && faceIds.contains(cFaceId)) {
            colors->InsertNextTypedTuple(highlight);
        } else if (hasSelection) {
            unsigned char base[3];
            hsvColor(cFaceId, base);
            unsigned char dimmed[3] = {
                static_cast<unsigned char>(base[0] * 0.35),
                static_cast<unsigned char>(base[1] * 0.35),
                static_cast<unsigned char>(base[2] * 0.35)
            };
            colors->InsertNextTypedTuple(dimmed);
        } else {
            unsigned char rgb[3];
            hsvColor(cFaceId, rgb);
            colors->InsertNextTypedTuple(rgb);
        }
    }

    previewPolyData_->GetCellData()->SetScalars(colors);
    previewPolyData_->Modified();

    // Re-apply clip so the new colors are visible through the clip pipeline
    applyClipToMapper();
#else
    (void)faceIds;
#endif
}

// ── Picking ───────────────────────────────────────────────────────────

#ifdef USE_VTK
void GeometryImportDialog::setupPickerCallback() {
    auto callback = vtkSmartPointer<vtkCallbackCommand>::New();
    callback->SetClientData(this);
    callback->SetCallback([](vtkObject* caller, unsigned long /*eventId*/,
                             void* clientData, void* /*callData*/) {
        auto* self = static_cast<GeometryImportDialog*>(clientData);
        auto* interactor = static_cast<vtkRenderWindowInteractor*>(caller);

        int* clickPos = interactor->GetEventPosition();
        self->cellPicker_->Pick(clickPos[0], clickPos[1], 0,
                                self->previewRenderer_);

        vtkIdType cellId = self->cellPicker_->GetCellId();
        if (cellId >= 0) {
            self->onCellPicked(cellId);
        }
    });

    auto* interactor = previewWidget_->renderWindow()->GetInteractor();
    if (interactor) {
        interactor->AddObserver(vtkCommand::LeftButtonPressEvent, callback);
    }
}

void GeometryImportDialog::onCellPicked(vtkIdType cellId) {
    if (!cellFaceIds_ || cellId < 0 ||
        cellId >= cellFaceIds_->GetNumberOfTuples()) return;

    int pickedFaceId = cellFaceIds_->GetValue(cellId);
    int pickedBodyId = cellBodyIds_->GetValue(cellId);

    // Find the tree item that matches and select it
    ignoreTreeSelection_ = true;

    if (isBodyMode()) {
        for (int i = 0; i < faceTree_->topLevelItemCount(); i++) {
            auto item = faceTree_->topLevelItem(i);
            if (item->data(0, Qt::UserRole + 1).toInt() == pickedBodyId) {
                faceTree_->setCurrentItem(item);
                break;
            }
        }
        // Highlight all faces of this body
        if (bodyToFaces_.contains(pickedBodyId)) {
            highlightInPreview(bodyToFaces_[pickedBodyId]);
        }
    } else {
        for (int i = 0; i < faceTree_->topLevelItemCount(); i++) {
            auto item = faceTree_->topLevelItem(i);
            if (item->data(0, Qt::UserRole).toInt() == pickedFaceId) {
                faceTree_->setCurrentItem(item);
                break;
            }
        }
        highlightInPreview({pickedFaceId});
    }

    ignoreTreeSelection_ = false;
}
#endif

// ── Tree selection → 3D highlight ─────────────────────────────────────

void GeometryImportDialog::onTreeSelectionChanged() {
    if (ignoreTreeSelection_) return;

    QSet<int> faceIds = getSelectedFaceIds();
    highlightInPreview(faceIds);
}

// ── Assignment actions ────────────────────────────────────────────────

void GeometryImportDialog::onAssignVoltage() {
    auto items = faceTree_->selectedItems();
    if (items.isEmpty()) {
        QMessageBox::information(this, "No Selection",
            "Select a face or body first (click in 3D or in the tree).");
        return;
    }

    auto item = items.first();
    QSet<int> faceIds = getSelectedFaceIds();
    if (faceIds.isEmpty()) return;

    bool ok;
    double defV = spinVoltage_->value();
    // If editing, use the existing voltage as default
    if (faceIds.size() == 1 && faceVoltages_.contains(*faceIds.begin()))
        defV = faceVoltages_[*faceIds.begin()];

    double voltage = QInputDialog::getDouble(this, "Set Voltage",
        QString("Voltage for %1 (V):").arg(item->text(0)),
        defV, -50000, 50000, 0, &ok);

    if (ok) {
        for (int fid : faceIds) {
            faceVoltages_[fid] = voltage;
        }
        item->setText(1, "Boundary");
        item->setText(2, QString::number(voltage, 'f', 0) + " V");
        item->setForeground(1, QBrush(QColor("#FF9800")));
    }
}

void GeometryImportDialog::onSetSource() {
    auto items = faceTree_->selectedItems();
    if (items.isEmpty()) {
        QMessageBox::information(this, "No Selection",
            "Select a face or body first (click in 3D or in the tree).");
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
    QSet<int> faceIds = getSelectedFaceIds();
    // Use the first face of the set as the source identifier
    sourceFaceId_ = faceIds.isEmpty() ? -1 : *faceIds.begin();
    item->setText(1, "Source");
    item->setForeground(1, QBrush(QColor("#9C27B0")));
}

} // namespace BEMCS
