#pragma once
#include <QWidget>
#include <QColor>
#include <vector>
#include <cstdint>

namespace BEMCS {

// ============================================================================
// Erosion Map Widget
//
// 2D heatmap of cumulative groove depth on the accel grid's downstream face.
// Every (x, y) column contributes a value, so the display is statistically
// much smoother than a 1D slice — no need to wait for ions to land exactly
// on a centre-line.
// ============================================================================
class ErosionMapWidget : public QWidget {
    Q_OBJECT
public:
    explicit ErosionMapWidget(QWidget* parent = nullptr);

    void setData(const std::vector<double>& depth_um,    // ny * nx, iy*nx + ix
                 const std::vector<uint8_t>& inAccel,    // same layout, 1 if accel column
                 int nx, int ny,
                 double Lx_mm, double Ly_mm,
                 double maxDepth_um);
    void clear();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    static QColor colormap(double t);  // jet-like, t in [0,1]

    std::vector<double>  depth_;
    std::vector<uint8_t> inAccel_;
    int nx_ = 0, ny_ = 0;
    double Lx_ = 0.0, Ly_ = 0.0;
    double peakDepth_ = 0.0;  // monotonic: never shrinks so the colour scale stays stable
};

} // namespace BEMCS
