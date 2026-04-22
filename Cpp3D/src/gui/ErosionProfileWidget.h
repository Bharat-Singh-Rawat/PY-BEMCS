#pragma once
#include <QWidget>
#include <vector>

namespace BEMCS {

// ============================================================================
// Erosion Profile Widget
//
// Lightweight 1D line plot that shows cumulative groove depth along the
// acceleration grid's downstream face in both the X and Y directions.
// Uses QPainter directly — no extra charting dependency needed.
// ============================================================================
class ErosionProfileWidget : public QWidget {
    Q_OBJECT
public:
    explicit ErosionProfileWidget(QWidget* parent = nullptr);

    // Feed fresh profile data. Both vectors per axis must be the same length.
    void setData(const std::vector<double>& x_coord_mm,
                 const std::vector<double>& x_depth_um,
                 const std::vector<double>& y_coord_mm,
                 const std::vector<double>& y_depth_um);
    void clear();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::vector<double> xCoord_, xDepth_;
    std::vector<double> yCoord_, yDepth_;
    double peakDepth_ = 0.0;  // tracks all-time max so y-axis never shrinks
};

} // namespace BEMCS
