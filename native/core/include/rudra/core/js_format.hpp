#pragma once
// Numbers as the browser Studio's JavaScript prints them, for output that must
// equal the page's byte for byte (the session's params(), the scopes' SVG).

#include <string>

namespace rudra {

// Number.prototype.toString / String(x): the shortest digits that read back
// to the same double, fixed between 1e-7 and 1e21, exponent form outside.
std::string js_number(double v);

// Number.prototype.toFixed(digits): the exact decimal value of the double
// rounded to `digits` places, halves up (away from zero); String(x) at or
// above 1e21.
std::string js_to_fixed(double v, int digits);

// Math.round: the nearest integer, halves toward +infinity.
double js_round(double v);

// The page's fmt(n, d): toLocaleString("en-US") with d fraction digits, so
// halves away from zero on the exact value and thousands grouped ("1,148").
// "—" for a number that is not finite.
std::string js_fmt(double v, int digits);
// The page's signed(n, d): "+" or U+2212 and fmt of the magnitude.
std::string js_signed(double v, int digits);
// The page's nitsLabel(v): "2k" from 2000, "0.05" from 0.05.
std::string nits_label(double v);

}  // namespace rudra
