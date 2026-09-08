#ifndef VAYU_CORE_OPS_HPP
#define VAYU_CORE_OPS_HPP

#include <xtensor/containers/xarray.hpp>
#include <utility>

// 1. Accelerated Horizontal Wind Vector Interpolation
void s2thta_kernel(const xt::xarray<double>& pthta, 
                   const xt::xarray<double>& plevs, 
                   const xt::xarray<double>& spres, 
                   const xt::xarray<double>& psfc, 
                   const xt::xarray<double>& ssfc, 
                   xt::xarray<double>& sthta);

// 2. Accelerated Fused Isentropic Coordinate Pressure Tracker
std::pair<xt::xarray<double>, xt::xarray<double>> execute_p2thta_fused_core(
    const xt::xarray<double>& thta_grid,
    const xt::xarray<double>& plevs,
    const xt::xarray<double>& potsfc,
    const xt::xarray<double>& psfc,
    const xt::xarray<double>& thtap_cleaned,
    double kappa, double epsln, int nmax, double p0_val, double missing_val);

#endif
