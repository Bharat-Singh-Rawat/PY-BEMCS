#pragma once
#include <cmath>
#include <iostream>

namespace BEMCS {

// Lightweight 3D vector for particle positions & velocities
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s)      const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s)      const { return {x / s, y / s, z / s}; }

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(double s)      { x *= s; y *= s; z *= s; return *this; }

    double dot(const Vec3& o)   const { return x * o.x + y * o.y + z * o.z; }
    double mag2()               const { return x * x + y * y + z * z; }
    double mag()                const { return std::sqrt(mag2()); }

    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y,
                z * o.x - x * o.z,
                x * o.y - y * o.x};
    }

    Vec3 normalized() const {
        double m = mag();
        return (m > 0) ? (*this / m) : Vec3{};
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec3& v) {
        os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

} // namespace BEMCS
