#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <cmath>
#include <algorithm>
#include <utility>
#include "include/vayu_core_ops.hpp"

const double HUNT_TOL = 0.001;

std::pair<xt::xarray<double>, xt::xarray<double>> execute_p2thta_fused_core(
    const xt::xarray<double>& thta_grid,
    const xt::xarray<double>& plevs,
    const xt::xarray<double>& potsfc,
    const xt::xarray<double>& psfc,
    const xt::xarray<double>& thtap_cleaned,
    double kappa, double epsln, int nmax, double p0_val, double missing_val)
{
    const size_t kthta = thta_grid.shape(0);
    const size_t nj    = potsfc.shape(0);
    const size_t ni    = potsfc.shape(1);
    const size_t plvls = plevs.shape(0);

    xt::xarray<double> log_plevs = xt::log(plevs);
    xt::xarray<double> pthta = xt::zeros<double>({kthta, nj, ni});
    xt::xarray<double> dltdlp_out = xt::zeros<double>({kthta, nj, ni});

#pragma omp parallel for collapse(2)
    for (size_t j = 0; j < nj; ++j) {
        for (size_t i = 0; i < ni; ++i) {
            double sfc_p  = psfc(j, i);
            double sfc_th = potsfc(j, i);
            double top_th = thtap_cleaned(plvls - 1, j, i);

            for (size_t k = 0; k < kthta; ++k) {
                double target_th = thta_grid(k);

                // --- BASELINE OUT OF BOUNDS CONTROLS ---
                if (target_th < sfc_th) {
                    pthta(k, j, i) = missing_val;
                    continue;
                }
                if (target_th > top_th) {
                    pthta(k, j, i) = missing_val;
                    continue;
                }
                if (std::abs(target_th - sfc_th) < HUNT_TOL) {
                    pthta(k, j, i) = sfc_p;
                    continue;
                }

                double p_down = 0.0, p_up = 0.0;
                double pot_down = 0.0, pot_up = 0.0;
                double alogp_down = 0.0, alogp_up = 0.0;
                bool layer_found = false;

                // --- BRANCH 1: SURFACE CONTACT ---
                if (target_th < thtap_cleaned(0, j, i)) {
                    p_down = sfc_p;
                    pot_down = sfc_th;
                    alogp_down = std::log(sfc_p);

                    bool c1 = (std::abs(sfc_p - plevs(0)) < HUNT_TOL);
                    pot_up   = c1 ? thtap_cleaned(1, j, i) : thtap_cleaned(0, j, i);
                    p_up     = c1 ? plevs(1) : plevs(0);
                    alogp_up = c1 ? log_plevs(1) : log_plevs(0);
                    layer_found = true;
                }

                // --- BRANCH 2: UPPER ATMOSPHERIC SWEEP ---
                if (!layer_found) {
                    for (size_t lvl = 1; lvl < plvls; ++lvl) {
                        if (target_th < thtap_cleaned(lvl, j, i)) {
                            if (sfc_th > thtap_cleaned(lvl - 1, j, i)) {
                                p_down = sfc_p;
                                pot_down = sfc_th;
                                alogp_down = std::log(sfc_p);

                                bool c3 = (std::abs(sfc_p - plevs(lvl)) < 0.01);
                                size_t lvl_plus_1 = (lvl + 1 < plvls) ? (lvl + 1) : lvl;
                                
                                pot_up   = c3 ? thtap_cleaned(lvl_plus_1, j, i) : thtap_cleaned(lvl, j, i);
                                p_up     = c3 ? plevs(lvl_plus_1) : plevs(lvl);
                                alogp_up = c3 ? log_plevs(lvl_plus_1) : log_plevs(lvl);
                            } else {
                                p_down = plevs(lvl - 1);
                                pot_down = thtap_cleaned(lvl - 1, j, i);
                                alogp_down = log_plevs(lvl - 1);

                                p_up = plevs(lvl);
                                pot_up = thtap_cleaned(lvl, j, i);
                                alogp_up = log_plevs(lvl);
                            }
                            layer_found = true;
                            break;
                        }
                    }
                }

                if (!layer_found) {
                    pthta(k, j, i) = missing_val;
                    continue;
                }

                // --- NEWTON-RAPHSON CORE CORE ENGINE ---
                double tdwn = pot_down * std::pow(p_down / p0_val, kappa);
                double tup  = pot_up   * std::pow(p_up / p0_val, kappa);

                double ratio = tup / ((tdwn == 0.0) ? 1.0 : tdwn);
                double log_ratio_combined = std::log(ratio);
                double denom = alogp_up - alogp_down;
                
                double dltdlp = log_ratio_combined / ((denom == 0.0) ? 1.0 : denom);
                double interc = std::log(tup) - (dltdlp * alogp_up);
                dltdlp_out(k, j, i) = dltdlp;

                double alogp_0 = log_plevs(0);
                double solver_denom = (dltdlp - kappa == 0.0) ? 1.0 : (dltdlp - kappa);
                double p_guess = std::exp((std::log(target_th) - interc - kappa * alogp_0) / solver_denom);

                int iter_count = 0;
                double current_p = p_guess;

                while (iter_count < nmax) {
                    double log_pthta = std::log(current_p);
                    double t1 = std::exp(dltdlp * log_pthta + interc);
                    double resid = current_p - p0_val * std::pow(t1 / target_th, 1.0 / kappa);

                    if (std::abs(resid) < epsln) break;

                    double thta1 = t1 * std::pow(p0_val / current_p, kappa);
                    double f_val = target_th - thta1;
                    double df_dp = (kappa - dltdlp) * std::pow(p0_val / current_p, kappa) * 
                                   std::exp(interc + (dltdlp - 1.0) * log_pthta);

                    double p1 = current_p - f_val / ((df_dp == 0.0) ? 1.0 : df_dp);

                    if (p1 <= p_down) {
                        if (p1 >= p_up) {
                            current_p = p1;
                        } else break;
                    } else break;
                    iter_count++;
                }
                pthta(k, j, i) = current_p;
            }
        }
    }
    return std::make_pair(pthta, dltdlp_out);
}
