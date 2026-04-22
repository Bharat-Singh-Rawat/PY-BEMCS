#include "gui/ErosionProfileWidget.h"
#include <QPainter>
#include <QPaintEvent>
#include <QFont>
#include <algorithm>

namespace BEMCS {

ErosionProfileWidget::ErosionProfileWidget(QWidget* parent)
    : QWidget(parent) {
    setMinimumSize(320, 220);
    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor("#1a1a2e"));
    setPalette(pal);
}

void ErosionProfileWidget::setData(
    const std::vector<double>& x_coord_mm,
    const std::vector<double>& x_depth_um,
    const std::vector<double>& y_coord_mm,
    const std::vector<double>& y_depth_um)
{
    xCoord_ = x_coord_mm;
    xDepth_ = x_depth_um;
    yCoord_ = y_coord_mm;
    yDepth_ = y_depth_um;

    for (double v : xDepth_) peakDepth_ = std::max(peakDepth_, v);
    for (double v : yDepth_) peakDepth_ = std::max(peakDepth_, v);

    update();
}

void ErosionProfileWidget::clear() {
    xCoord_.clear(); xDepth_.clear();
    yCoord_.clear(); yDepth_.clear();
    peakDepth_ = 0.0;
    update();
}

void ErosionProfileWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    const int W = width();
    const int H = height();
    p.fillRect(rect(), QColor("#1a1a2e"));

    const int ml = 60, mr = 18, mt = 32, mb = 46;
    const int pw = W - ml - mr;
    const int ph = H - mt - mb;
    if (pw <= 20 || ph <= 20) return;

    // Data ranges
    const bool haveX = xCoord_.size() >= 2;
    const bool haveY = yCoord_.size() >= 2;
    const bool haveData = haveX || haveY;

    double x_min = 0.0, x_max = 1.0;
    if (haveX) x_max = std::max(x_max, xCoord_.back());
    if (haveY) x_max = std::max(x_max, yCoord_.back());

    // Give the plot a little vertical headroom; floor at 1 μm so axes don't
    // collapse before any erosion has happened.
    const double y_min = 0.0;
    const double y_max = std::max(peakDepth_ * 1.1, 1.0);

    // Plot box
    p.setPen(QPen(QColor("#555"), 1));
    p.drawRect(ml, mt, pw, ph);

    // Interior gridlines
    p.setPen(QPen(QColor("#2e2e48"), 1, Qt::DotLine));
    for (int i = 1; i < 5; i++) {
        int yl = mt + i * ph / 5;
        p.drawLine(ml + 1, yl, ml + pw - 1, yl);
        int xl = ml + i * pw / 5;
        p.drawLine(xl, mt + 1, xl, mt + ph - 1);
    }

    // Tick labels
    p.setPen(QColor("#bbb"));
    p.setFont(QFont("Monospace", 8));
    for (int i = 0; i <= 5; i++) {
        double xv = x_min + i * (x_max - x_min) / 5.0;
        int xl = ml + i * pw / 5;
        p.drawText(xl - 16, mt + ph + 14, QString::number(xv, 'f', 2));

        double yv = y_max - i * (y_max - y_min) / 5.0;
        int yl = mt + i * ph / 5;
        p.drawText(6, yl + 4, QString::number(yv, 'f', 1));
    }

    // Axis labels
    p.setPen(QColor("#ddd"));
    p.setFont(QFont("Sans", 9));
    p.drawText(ml + pw / 2 - 60, H - 8, "Transverse coord [mm]");

    p.save();
    p.translate(14, mt + ph / 2 + 45);
    p.rotate(-90);
    p.drawText(0, 0, "Erosion depth [\xce\xbcm]");  // UTF-8 'µ'
    p.restore();

    // Title
    p.setPen(QColor("#e0e0e0"));
    p.setFont(QFont("Sans", 9, QFont::Bold));
    p.drawText(ml, mt - 14, "Accel Grid Erosion (downstream face)");

    // Lines
    auto drawSeries = [&](const std::vector<double>& c,
                          const std::vector<double>& d,
                          const QColor& col) {
        if (c.size() < 2) return;
        p.setPen(QPen(col, 2));
        QPoint prev;
        for (size_t i = 0; i < c.size(); i++) {
            double fx = (c[i] - x_min) / (x_max - x_min);
            double fy = (d[i] - y_min) / (y_max - y_min);
            int xp = ml + static_cast<int>(fx * pw);
            int yp = mt + static_cast<int>((1.0 - fy) * ph);
            QPoint cur(xp, yp);
            if (i > 0) p.drawLine(prev, cur);
            prev = cur;
        }
    };

    const QColor colX("#4fc3f7");
    const QColor colY("#ff8a65");
    drawSeries(xCoord_, xDepth_, colX);
    drawSeries(yCoord_, yDepth_, colY);

    // Legend (top-right inside plot area)
    p.setFont(QFont("Sans", 8));
    int lx = ml + pw - 150;
    int ly = mt + 8;
    p.setPen(QPen(colX, 2));
    p.drawLine(lx, ly, lx + 18, ly);
    p.setPen(QColor("#ddd"));
    p.drawText(lx + 24, ly + 4, "X  (y = centre)");

    p.setPen(QPen(colY, 2));
    p.drawLine(lx, ly + 14, lx + 18, ly + 14);
    p.setPen(QColor("#ddd"));
    p.drawText(lx + 24, ly + 18, "Y  (x = centre)");

    if (!haveData) {
        p.setPen(QColor("#777"));
        p.setFont(QFont("Sans", 10));
        p.drawText(rect(), Qt::AlignCenter,
                   "Build domain and run the simulation to populate.");
    }
}

} // namespace BEMCS
