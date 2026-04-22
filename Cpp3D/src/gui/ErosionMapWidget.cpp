#include "gui/ErosionMapWidget.h"
#include <QPainter>
#include <QPaintEvent>
#include <QImage>
#include <QFont>
#include <algorithm>
#include <cmath>

namespace BEMCS {

ErosionMapWidget::ErosionMapWidget(QWidget* parent)
    : QWidget(parent) {
    setMinimumSize(340, 260);
    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor("#1a1a2e"));
    setPalette(pal);
}

void ErosionMapWidget::setData(
    const std::vector<double>& depth_um,
    const std::vector<uint8_t>& inAccel,
    int nx, int ny,
    double Lx_mm, double Ly_mm,
    double maxDepth_um)
{
    depth_    = depth_um;
    inAccel_  = inAccel;
    nx_       = nx;
    ny_       = ny;
    Lx_       = Lx_mm;
    Ly_       = Ly_mm;
    peakDepth_ = std::max(peakDepth_, maxDepth_um);
    update();
}

void ErosionMapWidget::clear() {
    depth_.clear();
    inAccel_.clear();
    nx_ = ny_ = 0;
    Lx_ = Ly_ = 0.0;
    peakDepth_ = 0.0;
    update();
}

QColor ErosionMapWidget::colormap(double t) {
    // Jet-like: blue → cyan → green → yellow → red, piecewise linear
    t = std::clamp(t, 0.0, 1.0);
    double r, g, b;
    if (t < 0.25) {
        double u = t / 0.25;           // 0 → 1
        r = 0.0; g = u; b = 1.0;
    } else if (t < 0.50) {
        double u = (t - 0.25) / 0.25;
        r = 0.0; g = 1.0; b = 1.0 - u;
    } else if (t < 0.75) {
        double u = (t - 0.50) / 0.25;
        r = u; g = 1.0; b = 0.0;
    } else {
        double u = (t - 0.75) / 0.25;
        r = 1.0; g = 1.0 - u; b = 0.0;
    }
    return QColor::fromRgbF(r, g, b);
}

void ErosionMapWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);  // crisp pixel edges

    const int W = width();
    const int H = height();
    p.fillRect(rect(), QColor("#1a1a2e"));

    // Layout: title, plot area, colorbar, axis labels
    const int ml = 58, mt = 34, mb = 46;
    const int cbw = 18;           // colorbar width
    const int cbGap = 16;         // gap between plot and colorbar
    const int cbLabelWidth = 54;  // space to the right of the colorbar for numbers
    const int mr = cbw + cbGap + cbLabelWidth;
    const int pw = W - ml - mr;
    const int ph = H - mt - mb;
    if (pw <= 20 || ph <= 20) return;

    // Title
    p.setPen(QColor("#e0e0e0"));
    p.setFont(QFont("Sans", 9, QFont::Bold));
    p.drawText(ml, mt - 14, "Accel Grid Erosion Depth (downstream face)");

    // Nothing to draw yet?
    const bool haveData = (nx_ > 0 && ny_ > 0 &&
                           !depth_.empty() &&
                           (int)depth_.size() == nx_ * ny_);
    if (!haveData) {
        p.setPen(QColor("#777"));
        p.setFont(QFont("Sans", 10));
        p.drawText(rect(), Qt::AlignCenter,
                   "Build domain and run the simulation to populate.");
        return;
    }

    // Colour scale: lock to peakDepth so newly-eroded regions don't rescale
    // the palette on every frame. Floor at 1 μm so we don't divide by zero
    // before any erosion has happened.
    const double scaleMax = std::max(peakDepth_, 1.0);

    // Render the raw nx_ × ny_ field into a QImage, then scale to plot area.
    QImage img(nx_, ny_, QImage::Format_RGB32);
    const QColor outside("#2a2a3e");  // dark gray for cells outside accel region
    for (int iy = 0; iy < ny_; iy++) {
        // QImage rows go top→bottom; flip so y_mm = 0 sits at the bottom
        QRgb* scan = reinterpret_cast<QRgb*>(img.scanLine(ny_ - 1 - iy));
        for (int ix = 0; ix < nx_; ix++) {
            size_t k = static_cast<size_t>(iy) * nx_ + ix;
            if (k >= inAccel_.size() || !inAccel_[k]) {
                scan[ix] = outside.rgb();
                continue;
            }
            double t = depth_[k] / scaleMax;
            scan[ix] = colormap(t).rgb();
        }
    }

    // Scale to the plot area (FAST transform = nearest-neighbour → visible voxels
    // at low resolution, same vibe as the potential slice viewer).
    QImage scaled = img.scaled(pw, ph, Qt::IgnoreAspectRatio,
                               Qt::FastTransformation);
    p.drawImage(ml, mt, scaled);

    // Frame around plot
    p.setPen(QPen(QColor("#555"), 1));
    p.drawRect(ml, mt, pw, ph);

    // Tick labels (x = transverse-mm, y = transverse-mm)
    p.setPen(QColor("#bbb"));
    p.setFont(QFont("Monospace", 8));
    for (int i = 0; i <= 5; i++) {
        double xv = i * Lx_ / 5.0;
        int xl = ml + i * pw / 5;
        p.drawText(xl - 16, mt + ph + 14, QString::number(xv, 'f', 2));

        double yv = (5 - i) * Ly_ / 5.0;
        int yl = mt + i * ph / 5;
        p.drawText(6, yl + 4, QString::number(yv, 'f', 2));
    }

    // Axis labels
    p.setPen(QColor("#ddd"));
    p.setFont(QFont("Sans", 9));
    p.drawText(ml + pw / 2 - 30, H - 8, "x [mm]");
    p.save();
    p.translate(14, mt + ph / 2 + 25);
    p.rotate(-90);
    p.drawText(0, 0, "y [mm]");
    p.restore();

    // ── Colorbar ────────────────────────────────────────────────────────
    const int cbx = ml + pw + cbGap;
    const int cby = mt;
    const int cbh = ph;

    // Paint the gradient strip directly (no QImage needed, it's 1D).
    for (int j = 0; j < cbh; j++) {
        double t = 1.0 - static_cast<double>(j) / (cbh - 1);  // top = max
        p.setPen(colormap(t));
        p.drawLine(cbx, cby + j, cbx + cbw - 1, cby + j);
    }
    p.setPen(QPen(QColor("#555"), 1));
    p.drawRect(cbx, cby, cbw, cbh);

    // Colorbar ticks: min, 25%, 50%, 75%, max
    p.setPen(QColor("#bbb"));
    p.setFont(QFont("Monospace", 8));
    for (int i = 0; i <= 4; i++) {
        double frac = 1.0 - i / 4.0;            // top → bottom
        double val  = frac * scaleMax;
        int ty = cby + i * cbh / 4;
        p.setPen(QColor("#555"));
        p.drawLine(cbx + cbw, ty, cbx + cbw + 3, ty);
        p.setPen(QColor("#bbb"));
        p.drawText(cbx + cbw + 5, ty + 4, QString::number(val, 'f', 2));
    }

    // Colorbar unit label
    p.setPen(QColor("#ddd"));
    p.setFont(QFont("Sans", 8));
    p.drawText(cbx - 2, cby - 6, "depth [\xce\xbcm]");  // UTF-8 'µ'
}

} // namespace BEMCS
