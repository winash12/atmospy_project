#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <cmath>
#include <algorithm>
#include <vector>
#include "include/vayu_core_ops.hpp"


const double TOL = 0.01;

void s2thta_kernel(const xt::xarray<double>& pthta, 
                   const xt::xarray<double>& plevs, 
                   const xt::xarray<double>& spres,
                   const xt::xarray<double>& psfc,
                   const xt::xarray<double>& ssfc,
                   xt::xarray<double>& sthta) {
    
    auto shape = pthta.shape();
    const int kout = shape[0];
    const int nj = shape[1];
    const int ni = shape[2];
    const int plvls = plevs.size();

    // Pre-calculate isobaric log-ratios to match your lnpu1p/lnpu2p
    std::vector<double> lnpu1p(plvls - 1);
    std::vector<double> lnpu2p(plvls - 2);
    for(int i=0; i < plvls-1; ++i) lnpu1p[i] = std::log(plevs[i+1] / plevs[i]);
    for(int i=0; i < plvls-2; ++i) lnpu2p[i] = std::log(plevs[i+2] / plevs[i]);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            double sfc_p = psfc(j, i);
            double sfc_s = ssfc(j, i);

            for (int k = 0; k < kout; ++k) {
                double target_p = pthta(k, j, i);
                
                // --- IDENTITY CHECKS ---
                if (target_p <= 0) { sthta(k, j, i) = -9999.0; continue; }
                if (std::abs(target_p - sfc_p) < TOL) { sthta(k, j, i) = sfc_s; continue; }
                
                bool matched = false;
                for (int l = 0; l < plvls; ++l) {
                    if (std::abs(target_p - plevs[l]) < TOL) {
                        sthta(k, j, i) = spres(l, j, i);
                        matched = true; break;
                    }
                }
                if (matched) continue;

                double pdwn, pmid, pup, sdwn, smid, sup, l12, l13, l23;

                // BRANCH 1: Surface
                if (target_p > plevs[0]) {
                    pdwn = sfc_p; sdwn = sfc_s;
                    bool c1 = (std::abs(sfc_p - plevs[0]) < TOL);
                    pmid = c1 ? plevs[0] : plevs[1];
                    pup  = c1 ? plevs[1] : plevs[2];
                    smid = c1 ? spres(0, j, i) : spres(1, j, i);
                    sup  = c1 ? spres(1, j, i) : spres(2, j, i);
                    l12  = c1 ? lnpu1p[0] : lnpu1p[1];
                    l13  = c1 ? std::log(pup / pdwn) : lnpu2p[0];
                    l23  = c1 ? std::log(pmid / pdwn) : lnpu1p[0];
                } 
                // BRANCH 2: Top Cap
                else if (target_p < plevs[plvls-1]) {
                    pdwn = plevs[plvls-3]; pmid = plevs[plvls-2]; pup = plevs[plvls-1];
                    sdwn = spres(plvls-3, j, i); smid = spres(plvls-2, j, i); sup = spres(plvls-1, j, i);
                    l12 = lnpu1p[plvls-2]; l13 = lnpu2p[plvls-3]; l23 = lnpu1p[plvls-3];
                }
                // BRANCH 3: Internal (The F77 "100 Continue" loop)
                else {
                    int l_idx = 1;
                    // Safety: Stop at plvls-3 to ensure l_idx+2 is always safe
                    for (int l = 1; l < plvls - 2; ++l) {
                        if (target_p > plevs[l]) { l_idx = l; break; }
                    }

                    if (sfc_p < plevs[l_idx-1]) {
                        pdwn = sfc_p; sdwn = sfc_s;
                        bool c3 = (std::abs(sfc_p - plevs[l_idx]) < 0.001); // The specific F77 quirk
                        pmid = c3 ? plevs[l_idx] : plevs[l_idx+1];
                        pup  = c3 ? plevs[l_idx+1] : plevs[l_idx+2];
                        smid = c3 ? spres(l_idx, j, i) : spres(l_idx+1, j, i);
                        sup  = c3 ? spres(l_idx+1, j, i) : spres(l_idx+2, j, i);
                        l12  = c3 ? lnpu1p[l_idx] : lnpu1p[l_idx+1];
                        l13  = c3 ? std::log(pup / pdwn) : lnpu2p[l_idx];
                        l23  = c3 ? std::log(pmid / pdwn) : lnpu1p[l_idx];
                    } else {
                        pdwn = plevs[l_idx-1]; pmid = plevs[l_idx]; pup = plevs[l_idx+1];
                        sdwn = spres(l_idx-1, j, i); smid = spres(l_idx, j, i); sup = spres(l_idx+1, j, i);
                        l12 = lnpu1p[l_idx]; l13 = lnpu2p[l_idx-1]; l23 = lnpu1p[l_idx-1];
                    }
                }

                // QUADRATIC WEIGHTING
                double log_tp_pm = std::log(target_p / pmid);
                double log_tp_pp = std::log(target_p / pup);
                double log_tp_pd = std::log(target_p / pdwn);

                sthta(k, j, i) = ((log_tp_pm * log_tp_pp) / (l23 * l13)) * sdwn +
                                 ((-log_tp_pd * log_tp_pp) / (l23 * l12)) * smid +
                                 ((log_tp_pd * log_tp_pm) / (l13 * l12)) * sup;
            }
        }
    }
}






