#ifndef UTIL_H
#define UTIL_H


namespace util {
  template <typename T> bool is_not_nan(const T& x) { return x == x; }
  template <typename T> bool is_finite(const T& x) { return is_not_nan(x - x); }
}


#endif
