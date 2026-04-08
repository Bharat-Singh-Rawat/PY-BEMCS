#pragma once
#include "core/Particle.h"
#include <QWidget>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QPushButton>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QLabel>
#include <vector>

namespace BEMCS {

// ============================================================================
// Control Panel Widget - Left sidebar with all simulation parameters
// ============================================================================
class ControlPanel : public QWidget {
    Q_OBJECT

public:
    explicit ControlPanel(QWidget* parent = nullptr);

    // Extract current parameters from UI
    SimParams getParams() const;

    // Update diagnostics display
    void updateDiagnostics(int iteration, int ionCount, int electronCount,
                           double divergence, double saddlePot,
                           double meanEnergy, const std::vector<double>& gridTemps);

signals:
    void buildDomainRequested();
    void startStopToggled(bool running);
    void importGeometryRequested();
    void meshSettingsRequested();
    void resetRequested();
    void exportDataRequested();

    // View signals
    void viewXYRequested();
    void viewXZRequested();
    void viewIsoRequested();
    void showParticlesChanged(bool);
    void showPotentialChanged(bool);
    void showTemperatureChanged(bool);
    void showDamageChanged(bool);
    void slicePlaneChanged(int);
    void slicePositionChanged(double);

private:
    void setupUI();
    QHBoxLayout* createSpinRow(const QString& label, QDoubleSpinBox*& spin,
                                double min, double max, double val,
                                double step, int decimals);

    // Grid optic widgets
    struct GridWidget {
        QGroupBox* group;
        QDoubleSpinBox* voltage;
        QDoubleSpinBox* thickness;
        QDoubleSpinBox* gap;
        QDoubleSpinBox* radius;
        QDoubleSpinBox* chamfer;
    };

    void addGridUI(double v, double t, double gap, double r, double cham);
    void removeGridUI();

    std::vector<GridWidget> gridWidgets_;
    QVBoxLayout* gridsLayout_;

    // Plasma params
    QDoubleSpinBox* spinPlasmaDens_;
    QDoubleSpinBox* spinTeUp_;
    QDoubleSpinBox* spinTi_;
    QDoubleSpinBox* spinTn_;
    QDoubleSpinBox* spinNeutralDens_;
    QDoubleSpinBox* spinAccel_;
    QDoubleSpinBox* spinThresh_;

    // RF
    QCheckBox* chkRF_;
    QComboBox* comboRFGrid_;
    QDoubleSpinBox* spinRFFreq_;
    QDoubleSpinBox* spinRFAmp_;

    // Neutralizer
    QSpinBox* spinNeutRate_;
    QDoubleSpinBox* spinNeutTemp_;

    // Domain
    QDoubleSpinBox* spinLx_, *spinLy_, *spinLz_;
    QDoubleSpinBox* spinDx_;

    // Simulation mode
    QComboBox* comboSimMode_;

    // Diagnostics labels
    QLabel* lblIteration_;
    QLabel* lblIonCount_;
    QLabel* lblElecCount_;
    QLabel* lblDivergence_;
    QLabel* lblSaddlePot_;
    QLabel* lblMeanEnergy_;
    QLabel* lblGridTemps_;

    // Control buttons
    QPushButton* btnStartStop_;
    bool isRunning_ = false;
};

} // namespace BEMCS
