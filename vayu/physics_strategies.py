import numpy as np
from strategy_interface import GenericDomainStrategy

class LegacySAStrategy(GenericDomainStrategy):
    """Vectorized baseline Superadiabatic (SA) override with strict F77 structural parity."""

    def execute(self, *args, **kwargs):
        psfc, plevs, potsfc, thtap = args
        plvls = len(plevs)
        
        # Enforce pure float64 arrays
        tht_mod = np.array(thtap, dtype=np.float64, copy=True)
        psfc = np.asarray(psfc, dtype=np.float64)
        potsfc = np.asarray(potsfc, dtype=np.float64)
        plevs = np.asarray(plevs, dtype=np.float64)

        # Fortran STRAT_IDX = (PLVLS / 2) + 1 (1-based index) -> index (PLVLS // 2) in 0-based Python
        strat_idx = plvls // 2
        thtahi = np.float64(tht_mod[strat_idx, 0, 0])

        plevs_3d = plevs[:, np.newaxis, np.newaxis]

        # --- LEVEL k = 0 (Fortran KIN = 1 Base Case) ---
        underground_k0 = psfc > plevs_3d[0]
        violates_k0 = underground_k0 & (tht_mod[0] <= potsfc)
        tht_mod[0] = np.where(violates_k0, potsfc + np.float64(0.01), tht_mod[0])

        # --- SEQUENTIAL VERTICAL PASS (Fortran KIN = 2 to PLVLS) ---
        for k in range(1, plvls):
            underground_mask = psfc > plevs_3d[k]

            if np.any(underground_mask):
                # Branch A: Sit between ground and previous level
                is_branch_a = underground_mask & (psfc < plevs_3d[k-1])
                
                # Branch B: Deep underground (STRICT Fortran 'ELSE' complement)
                is_branch_b = underground_mask & (~is_branch_a)

                # Violations evaluated strictly per branch
                violates_a = is_branch_a & (tht_mod[k] < potsfc)
                violates_b = is_branch_b & (tht_mod[k] <= tht_mod[k-1])

                # Apply non-overlapping modifications
                tht_mod[k] = np.where(violates_a, potsfc + np.float64(0.01), tht_mod[k])
                tht_mod[k] = np.where(violates_b, tht_mod[k-1] + np.float64(0.01), tht_mod[k])

            # Stratospheric Ceiling Tracker Pass
            if k >= strat_idx:
                max_level_val = np.max(tht_mod[k])
                if max_level_val > thtahi:
                    thtahi = np.float64(max_level_val)

        return tht_mod, thtahi

class Moore1993Strategy(GenericDomainStrategy):
    """
    James T. Moore (1993) Local Column Mixer with fused THTAHI tracking.
    
    Replaces non-physical legacy upward nudging with a local, mass/energy-conserving 
    thermal mixing scheme (midpoint adjustment).
    
    Fully dynamic: adaptively resolves vertical grids with 16, 22, 37, or 137 levels.
    """

    def execute(self, *args, **kwargs):
        psfc, plevs, potsfc, thtap = args
        plvls = len(plevs)

        # Enforce pure float64 arrays for maximum numerical precision
        tht_mod = np.array(thtap, dtype=np.float64, copy=True)
        psfc = np.asarray(psfc, dtype=np.float64)
        potsfc = np.asarray(potsfc, dtype=np.float64)
        plevs = np.asarray(plevs, dtype=np.float64)

        # Inject custom epsilon from configuration (default: 0.005 K)
        epsln = np.float64(kwargs.get("MOORE_EPSILON", 0.005))
        max_sweeps = kwargs.get("NMAX", 10)

        # Dynamic stratospheric index: Finds first level <= 500 hPa (50000 Pa)
        # Fallback to plvls // 2 if plevs grid remains above 500 hPa
        strat_mask = plevs <= 50000.0
        if np.any(strat_mask):
            strat_idx = int(np.argmax(strat_mask))
        else:
            strat_idx = plvls // 2

        thtahi = np.float64(tht_mod[strat_idx, 0, 0])
        plevs_3d = plevs[:, np.newaxis, np.newaxis]

        # --- LEVEL k = 0 (Base Surface Correction) ---
        underground_k0 = psfc > plevs_3d[0]
        violates_k0 = underground_k0 & (tht_mod[0] <= potsfc)
        tht_mod[0] = np.where(violates_k0, potsfc + epsln, tht_mod[0])

        # --- ITERATIVE CONVERGENCE LOOP (Handles deep multi-layer inversions) ---
        for _ in range(max_sweeps):
            instability_found = False

            for k in range(1, plvls):
                underground_mask = psfc > plevs_3d[k]

                if np.any(underground_mask):
                    unstable_mask = underground_mask & (tht_mod[k] <= tht_mod[k - 1])

                    if np.any(unstable_mask):
                        instability_found = True

                        # Conserved thermal energy midpoint
                        midpoint = (tht_mod[k] + tht_mod[k - 1]) / 2.0

                        # Symmetrical energy-conserving split around midpoint
                        tht_mod[k - 1] = np.where(unstable_mask, midpoint - epsln, tht_mod[k - 1])
                        tht_mod[k]     = np.where(unstable_mask, midpoint + epsln, tht_mod[k])

            # Exit early if the whole vertical column has stabilized
            if not instability_found:
                break

        # --- STRATOSPHERIC CEILING TRACKER PASS ---
        for k in range(strat_idx, plvls):
            max_level_val = np.max(tht_mod[k])
            if max_level_val > thtahi:
                thtahi = np.float64(max_level_val)

        return tht_mod, thtahi
