#ifndef UTIL_H
#define UTIL_H

//#define likely(x) __builtin_expect(!!(x), true)
//#define unlikely(x) __builtin_expect(!!(x), false)

namespace util {
  template <typename T> bool is_not_nan(const T& x) { return x == x; }
  template <typename T> bool is_finite(const T& x) { return is_not_nan(x - x); }
}

#endif
