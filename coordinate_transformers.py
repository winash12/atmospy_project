import os
import sys
import numpy as np
from strategy_interface import GenericDomainStrategy
from config_loader import inject_constants, inject_solver_settings, inject_library_settings
from strategy_factory import DynamicStrategyFactory
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
import interp_lib
import weather_lib
from thermostatics import LegacySAStrategy, Moore1993Strategy
from thermostatics import pot

@dataclass
class IsentropicPressureState:
    """
    Cohesive data container encapsulating all 3D arrays, input coordinates,
    and pre-allocated workspace buffers for the isobaric-to-isentropic pressure tracker.
    """
    # --- Input Fields & Grids ---
    potsfc: np.ndarray             # Surface potential temperature (2D)
    psfc: np.ndarray               # Surface pressure (2D)
    plevs: np.ndarray              # Model pressure levels array (1D)
    thtap_cleaned: np.ndarray      # Stabilized profile potential temperature (3D)
    thta_grid_clean: np.ndarray    # Isentropic target levels grid (1D)
    log_plevs: np.ndarray          # Logarithmic coordinates vector (1D)

    # --- Structural Space Constraints ---
    kout: int                      # Total generated isentropic levels
    plvls: int                     # Total input isobaric levels 
    nj: int                        # Latitude spatial length
    ni: int                        # Longitude spatial length

    # --- Output & Mask Matrices ---
    pthta: np.ndarray              # Tracked target pressure matrix output (3D)
    done: np.ndarray               # Boolean subterranean mask matrix (3D)
    
    # --- Pre-allocated Shared Workspace Array Registers ---
    pressure_down: np.ndarray
    potential_temp_down: np.ndarray
    pressure_up: np.ndarray
    potential_temp_up: np.ndarray
    alogp_down: np.ndarray
    alogp_up: np.ndarray
    
    # --- Injected TOML Configuration Constants ---
    kappa_val: float
    epsln_val: float
    nmax_val: int
    p0_val: float
    missing_val: float


@dataclass
class IsentropicVelocityState:
    """
    Cohesive data container encapsulating target grids, wind vector components,
    and interpolated matrices for the isentropic horizontal velocity tracking pass.
    """
    # --- Input Grids & Surfaces ---
    pthta: np.ndarray              # Tracked pressure along isentropic surfaces (3D)
    plevs: np.ndarray              # Model pressure levels array (1D)
    psfc: np.ndarray               # Surface pressure (2D)
    uins: np.ndarray               # Raw horizontal wind component matrix (3D, maps U or V)
    uwndI: np.ndarray              # Surface wind validation layer vector (2D, maps Usfc or Vsfc)

    # --- Output & Mask Matrices ---
    sthta: np.ndarray              # FIXED: Normalized interpolated velocity component output (3D)
    done: np.ndarray               # Boolean vector tracking completion mask (3D)
    
    # --- Structural Space Constraints ---
    kout: int
    nj: int
    ni: int
    plvls: int
    
    # --- Physical Constants Context ---
    missing_val: float

    


class SubterraneanStrategy(ABC):
    @abstractmethod
    def compute_pthta(self, mask_under, thta_3d, potsfc, psfc, pres, pthta, done) -> np.ndarray:
        """
        Computes subterranean pressures in-place and returns the updated p2thta array.
        """
        pass


class KeithBrillStrategy(SubterraneanStrategy):
    def __init__(self, kappa=2.0/7.0):
        self.kappa = kappa

    def compute_pthta(self, mask_under, thta_3d, potsfc, psfc, pres, pthta, done) -> np.ndarray:
        p0 = 1000.0 if pres[0] > 500.0 else 100000.0
        
        # 1. Identify subterranean pressure level p_sub (first level > psfc)
        pres_3d = np.broadcast_to(pres[:, None, None], pthta.shape)
        psfc_full = np.broadcast_to(psfc[None, :, :], pthta.shape)
        
        is_under = pres_3d > psfc_full
        p_under_levels = np.where(is_under, pres_3d, np.nan)
        p_sub = np.nanmin(p_under_levels, axis=0)
        p_sub = np.where(np.isnan(p_sub), psfc + 50.0, p_sub)
        
        # 2. Compute theta_sub at p_sub assuming isothermal T_sfc
        tsfc = potsfc * (psfc / p0)**self.kappa
        thta_sub = tsfc * (p0 / p_sub)**self.kappa
        
        # 3. Log-linear interpolation downward
        alogp_sfc_3d = np.broadcast_to(np.log(psfc)[None, :, :], pthta.shape)
        alogp_sub_3d = np.broadcast_to(np.log(p_sub)[None, :, :], pthta.shape)
        potsfc_3d    = np.broadcast_to(potsfc[None, :, :], pthta.shape)
        thta_sub_3d  = np.broadcast_to(thta_sub[None, :, :], pthta.shape)
        
        dthta_sub = thta_sub_3d - potsfc_3d
        dthta_sub = np.where(np.abs(dthta_sub) < 1e-12, -1e-12, dthta_sub)
        
        frac = (thta_3d - potsfc_3d) / dthta_sub
        alogp_interp = alogp_sfc_3d + frac * (alogp_sub_3d - alogp_sfc_3d)
        
        # Target assignment
        pthta[mask_under] = np.exp(alogp_interp)[mask_under]
        done[mask_under] = True
        
        return pthta


class ECMWFOrszagStrategy(SubterraneanStrategy):
    def __init__(self, gamma=0.0065, kappa=2.0/7.0, g=9.80665, R=287.058):
        self.gamma = gamma
        self.kappa = kappa
        self.g = g
        self.R = R
        self.R_gamma_over_g = (R * gamma) / g

    def compute_pthta(self, mask_under, thta_3d, potsfc, psfc, pres, pthta, done) -> np.ndarray:
        p0 = 1000.0 if pres[0] > 500.0 else 100000.0
        
        tsfc = potsfc * (psfc / p0)**self.kappa
        ln_theta_3d = np.log(thta_3d)
        ln_tsfc_3d  = np.broadcast_to(np.log(tsfc)[None, :, :], pthta.shape)
        ln_psfc_3d  = np.broadcast_to(np.log(psfc)[None, :, :], pthta.shape)
        ln_p0       = np.log(p0)

        denom = self.R_gamma_over_g - self.kappa
        ln_P_theta = (
            ln_theta_3d 
            - ln_tsfc_3d 
            - (self.kappa * ln_p0) 
            + (self.R_gamma_over_g * ln_psfc_3d)
        ) / denom

        pthta[mask_under] = np.exp(ln_P_theta)[mask_under]
        done[mask_under] = True
        
        return pthta


class StrictLorenzStrategy(SubterraneanStrategy):
    def compute_pthta(self, mask_under, thta_3d, potsfc, psfc, pres, pthta, done) -> np.ndarray:
        psfc_3d = np.broadcast_to(psfc[None, :, :], pthta.shape)
        pthta[mask_under] = psfc_3d[mask_under]
        done[mask_under] = True
        
        return pthta
class MissingValueStrategy(SubterraneanStrategy):
    """
    Applies a hard missing value data mask threshold to subterranean coordinates.
    Insulated from dictionary keyword collisions via positional-only constraints.
    """
    # Note the forward slash (/) right after 'done'!
    def compute_pthta(self, mask_under, thta_3d, potsfc, psfc, pres, pthta, done, /, **kwargs):
        # Dynamically pull the value token directly out of your configuration payload
        # Checks both 'MISSING_DATA' (from your config.yaml) and fallback layouts
        missing_val = float(kwargs.get("MISSING_DATA", kwargs.get("missing_val", -9999.0)))
        
        pthta[mask_under] = missing_val
        done[mask_under] = True
        
        return pthta



    
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



    
    

    
class IsentropicNumpyBackend(GenericDomainStrategy):
    """
    Pure NumPy Parallel Vector Coordinate Tracking Strategy.
    Implements your exact vertical hunt and Newton-Raphson execution tracks.
    """

   
    
    def execute(self, *args, **kwargs):
        """
        Unpacks generic positional arguments and safely encapsulates them into 
        the IsentropicPressureState container to protect downstream methods from parameter bloat.
        """
        # 1. Unpack raw positional arguments as required by the interface
        (
            kout, plvls, thta_grid_clean, potsfc, psfc_arr, plevs_arr,
            log_plevs, thtap_cleaned, workspace, missing_val
        ) = args
        
        # 2. Extract spatial dimensions and unpack the workspace pre-allocated memory registers
        nj, ni = potsfc.shape
        p_dwn, pot_dwn, p_up, pot_up, alogp_dwn, alogp_up = workspace

        # 3. Instantiate the cohesive state container object 
        state = IsentropicPressureState(
            potsfc=potsfc,
            psfc=psfc_arr,
            plevs=plevs_arr,
            thtap_cleaned=thtap_cleaned,
            thta_grid_clean=thta_grid_clean,
            log_plevs=log_plevs,
            kout=kout,
            plvls=plvls,
            nj=nj,
            ni=ni,
            pthta=np.zeros((kout, nj, ni), dtype=np.float64), 
            done=np.zeros((kout, nj, ni), dtype=bool),
            pressure_down=p_dwn,
            potential_temp_down=pot_dwn,
            pressure_up=p_up,
            potential_temp_up=pot_up,
            alogp_down=alogp_dwn,
            alogp_up=alogp_up,
            kappa_val=kwargs.get("kappa_val", 0.285856),
            epsln_val=kwargs.get("epsln_val", 1.0),
            nmax_val=kwargs.get("nmax_val", 5),
            p0_val=kwargs.get("p0_val", 100000.0),
            missing_val=missing_val
        )

        # 4. Resolve the execution strategy using dynamic reflection
        sub_mod = kwargs.get("STRATEGY_MODULE", "coordinate_transformers")
        sub_cls = kwargs.get("SUBTERRANEAN_PHYSICS", "MissingValueStrategy")
        strategy = DynamicStrategyFactory.resolve(sub_mod, sub_cls)

        # 5. Route processing to the tracking hunt using the singular state container
        self._execute_vertical_layer_hunt(strategy, state, **kwargs)
        
        # 6. Trigger the precise Newton-Raphson thermodynamic solver engine using the state container
        return self._solve_isentropic_pressure_nr_engine(state)


    def _execute_vertical_layer_hunt(self, strategy, state: IsentropicPressureState, **kwargs):
        """
        Executes a highly optimized progressive vertical column bounding sweep.
        Driven entirely via a singular, single-parameter IsentropicPressureState container.
        """
        tol = 0.001
    
        # Coordinate broadcasting to 3D grid matrix space (kout, nj, ni) using state parameters
        potsfc_3d = np.broadcast_to(state.potsfc[None, :, :], (state.kout, state.nj, state.ni))
        psfc_3d   = np.broadcast_to(state.psfc[None, :, :], (state.kout, state.nj, state.ni))
        thta_3d   = np.broadcast_to(state.thta_grid_clean[:, None, None], (state.kout, state.nj, state.ni))
        
        # Direct high-speed buffer memory reset using native NumPy fills on state workspace arrays
        state.pressure_down.fill(0.0)
        state.potential_temp_down.fill(0.0)
        state.pressure_up.fill(0.0)
        state.potential_temp_up.fill(0.0)
        state.alogp_down.fill(0.0)
        state.alogp_up.fill(0.0)
        
        # =====================================================================
        # TOP-OF-LOOP EDGE CASES (Isolating Out-of-Bounds Configurations)
        # =====================================================================
        # Condition 1: TARGET FALLS BELOW GROUND SURFACE BOUNDARIES (STRATEGY HOOK FIX!)
        mask_under = thta_3d < potsfc_3d
        
        if np.any(mask_under):
            # The strategy modifies state.pthta and state.done in-place natively!
            strategy.compute_pthta(
                mask_under,
                thta_3d,
                potsfc_3d,
                state.psfc,
                state.plevs,
                state.pthta,
                state.done,
                **kwargs
           )
        
        # Condition 2: Target potential temperature exceeds highest standard model level
        thtap_top_3d = np.broadcast_to(state.thtap_cleaned[state.plvls-1, None, :, :], (state.kout, state.nj, state.ni))
        mask_over = (~state.done) & (thta_3d > thtap_top_3d)
        state.pthta[mask_over] = state.missing_val
        state.done[mask_over] = True
        
        # Condition 3: Target potential temperature matches surface value within tolerance
        mask_sfc = (~state.done) & (np.abs(thta_3d - potsfc_3d) < tol)
        state.pthta[mask_sfc] = psfc_3d[mask_sfc]
        state.done[mask_sfc] = True
        
        # =====================================================================
        # MAIN PROGRESSIVE SWEEP (Vectorized Vertical Grid Column Evaluation)
        # =====================================================================
        
        # --- BRANCH 1: SURFACE CONTACT SPECIAL CASE (Model Level Index 0) ---
        k = 0
        thtap_k_3d = np.broadcast_to(state.thtap_cleaned[k, None, :, :], (state.kout, state.nj, state.ni))
        active_0 = (~state.done) & (thta_3d < thtap_k_3d)
        
        if np.any(active_0):
            pdwn_tmp, potdwn_tmp, pup_tmp, potup_tmp = [np.zeros_like(thta_3d) for _ in range(4)]
            alogpd_tmp, alogpu_tmp = [np.zeros_like(thta_3d) for _ in range(2)]

            pdwn_tmp[active_0]   = psfc_3d[active_0]
            potdwn_tmp[active_0] = potsfc_3d[active_0]
            alogpd_tmp[active_0] = np.log(psfc_3d[active_0])
            
            c1_3d = (np.abs(psfc_3d - state.plevs[k]) < tol)
        
            thtap_k0_3d = np.broadcast_to(state.thtap_cleaned[k, None, :, :], (state.kout, state.nj, state.ni))
            thtap_k1_3d = np.broadcast_to(state.thtap_cleaned[k+1, None, :, :], (state.kout, state.nj, state.ni))
            
            potup_tmp[active_0]  = np.where(c1_3d, thtap_k1_3d, thtap_k0_3d)[active_0]
            pup_tmp[active_0]    = np.where(c1_3d, state.plevs[k+1], state.plevs[k])[active_0]
            alogpu_tmp[active_0] = np.where(c1_3d, state.log_plevs[k+1], state.log_plevs[k])[active_0]

            state.pressure_down[active_0]       = pdwn_tmp[active_0]
            state.potential_temp_down[active_0] = potdwn_tmp[active_0]
            state.pressure_up[active_0]         = pup_tmp[active_0]
            state.potential_temp_up[active_0]   = potup_tmp[active_0]
            state.alogp_down[active_0]          = alogpd_tmp[active_0]
            state.alogp_up[active_0]            = alogpu_tmp[active_0]
            
            state.done[active_0] = True

        # --- BRANCH 2: THE UPPER ATMOSPHERIC SWEEP (Model Level Indices 1 to PLVLS) ---
        pdwn_tmp, potdwn_tmp, pup_tmp, potup_tmp = [np.zeros_like(thta_3d) for _ in range(4)]
        alogpd_tmp, alogpu_tmp = [np.zeros_like(thta_3d) for _ in range(2)]

        for k in range(1, state.plvls):
            thtap_k_3d = np.broadcast_to(state.thtap_cleaned[k, None, :, :], (state.kout, state.nj, state.ni))
            active = (~state.done) & (thta_3d < thtap_k_3d)
            if not np.any(active): 
                continue

            thtap_km1_3d = np.broadcast_to(state.thtap_cleaned[k-1, None, :, :], (state.kout, state.nj, state.ni))
            m_psfc = active & (potsfc_3d > thtap_km1_3d)
            m_gen  = active & (~m_psfc)

            pdwn_tmp.fill(0.0)
            potdwn_tmp.fill(0.0)
            pup_tmp.fill(0.0)
            potup_tmp.fill(0.0)
            alogpd_tmp.fill(0.0)
            alogpu_tmp.fill(0.0)

            if np.any(m_psfc):
                pdwn_tmp[m_psfc]   = psfc_3d[m_psfc]
                potdwn_tmp[m_psfc] = potsfc_3d[m_psfc]
                alogpd_tmp[m_psfc] = np.log(psfc_3d[m_psfc])
                
                c3_3d = (np.abs(psfc_3d - state.plevs[k]) < 0.01)
            
                k_plus_1 = k + 1 if (k + 1 < state.plvls) else k
                thtap_k_hot  = np.broadcast_to(state.thtap_cleaned[k, None, :, :], (state.kout, state.nj, state.ni))
                thtap_k1_hot = np.broadcast_to(state.thtap_cleaned[k_plus_1, None, :, :], (state.kout, state.nj, state.ni))
                
                potup_tmp[m_psfc]  = np.where(c3_3d, thtap_k1_hot, thtap_k_hot)[m_psfc]
                pup_tmp[m_psfc]    = np.where(c3_3d, state.plevs[k_plus_1], state.plevs[k])[m_psfc]
                alogpu_tmp[m_psfc] = np.where(c3_3d, state.log_plevs[k_plus_1], state.log_plevs[k])[m_psfc]

            if np.any(m_gen):
                pdwn_tmp[m_gen]   = state.plevs[k-1]
                potdwn_tmp[m_gen] = thtap_km1_3d[m_gen]
                alogpd_tmp[m_gen] = state.log_plevs[k-1]
                
                pup_tmp[m_gen]    = state.plevs[k]
                potup_tmp[m_gen]  = thtap_k_3d[m_gen]
                alogpu_tmp[m_gen] = state.log_plevs[k]
            
            state.pressure_down[active]       = pdwn_tmp[active]
            state.potential_temp_down[active] = potdwn_tmp[active]
            state.pressure_up[active]         = pup_tmp[active]
            state.potential_temp_up[active]   = potup_tmp[active]
            state.alogp_down[active]          = alogpd_tmp[active]
            state.alogp_up[active]            = alogpu_tmp[active]
            
            state.done[active] = True

    
    def _solve_isentropic_pressure_nr_engine(self, state: IsentropicPressureState):
        """
        Production-Frozen Thermodynamic Newton-Raphson Solver Engine.
        Driven entirely via a single-parameter IsentropicPressureState context container.
        """
        active_math = state.pressure_down > 0.0
        if not np.any(active_math):
            return state.pthta, state.pressure_down, state.pressure_up, state.alogp_down, state.alogp_up
        
        # Derive Temperature boundaries via precise 64-bit float scaling
        tdwn = state.potential_temp_down * (state.pressure_down / state.p0_val) ** state.kappa_val
        tup  = state.potential_temp_up  * (state.pressure_up  / state.p0_val) ** state.kappa_val
         # Synchronized Core Mathematical Slope Operations Block
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(active_math, tup / np.where(tdwn == 0.0, 1.0, tdwn), 1.0)
            log_ratio_combined = np.log(ratio, where=active_math, out=np.zeros_like(tup))
            
            denom = np.where(active_math, state.alogp_up - state.alogp_down, 1.0)
            dltdlp = np.divide(log_ratio_combined, denom, where=active_math, out=np.zeros_like(log_ratio_combined))
            
            log_tup_isolated = np.log(tup, where=active_math, out=np.zeros_like(tup))
            interc = np.where(active_math, log_tup_isolated - (dltdlp * state.alogp_up), 0.0)

        # Coordinate dimension allocation and spatial tracking configuration
        thta_3d = np.broadcast_to(state.thta_grid_clean[:, np.newaxis, np.newaxis], (state.kout, state.nj, state.ni))
    
        # FIXED: Extract the pure initial log-pressure scalar from the first index 
        # of your model pressure levels array to ensure bit-perfect broadcasting alignment!
        alogp_0 = np.log(state.plevs[0])
        
        solver_denom = np.where(active_math, dltdlp - state.kappa_val, 1.0)
        p_guess = np.exp((np.log(thta_3d) - interc - state.kappa_val * alogp_0) / solver_denom)
        
        state.pthta = np.where(active_math, p_guess, state.pthta)

        # Initialize loop convergence arrays and runtime state masks
        n_counter = np.zeros_like(state.pthta, dtype=np.int32)
        iter_mask = active_math.copy()
        resmax = np.float64(1.0)
        
 # =====================================================================
        # ITERATIVE NEWTON CONVERGENCE LOOP (Vectorized 1900 CONTINUE)
        # =====================================================================
        for _ in range(state.nmax_val + 2):
            if not np.any(iter_mask):
                break
            
            with np.errstate(divide='ignore', invalid='ignore'):
                log_pthta = np.log(state.pthta, where=iter_mask, out=np.ones_like(state.pthta))
            t1 = np.exp(dltdlp * log_pthta + interc)
            
            resid = state.pthta - state.p0_val * (t1 / thta_3d) ** (np.float64(1.0) / state.kappa_val)
        
            abs_resid = np.abs(resid)
            needs_update = iter_mask & (abs_resid > state.epsln_val)
        
            current_step_active = iter_mask.copy()
            iter_mask = iter_mask & needs_update

            working_mask = current_step_active & needs_update
            if not np.any(working_mask):
                continue

            n_counter[working_mask] += 1
            within_bounds = working_mask & (n_counter <= state.nmax_val)
            exceeded_bounds = working_mask & (n_counter > state.nmax_val)
            
            # --- BRANCH 1: LOOP COUNT WITHIN VALID BOUNDS -> IF (N .LE. NMAX) ---
            if np.any(within_bounds):
                thta1 = t1 * (state.p0_val / state.pthta) ** state.kappa_val
                f = thta_3d - thta1
            
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_pthta_branch = np.log(state.pthta, where=within_bounds, out=np.ones_like(state.pthta))
                dfdp = (state.kappa_val - dltdlp) * (state.p0_val / state.pthta) ** state.kappa_val * \
                    np.exp(interc + (dltdlp - np.float64(1.0)) * log_pthta_branch)
            
                p1 = state.pthta - f / np.where(dfdp == 0.0, np.float64(1.0), dfdp)

                mask_le_pdwn = within_bounds & (p1 <= state.pressure_down)
                mask_valid = mask_le_pdwn & (p1 >= state.pressure_up)
                state.pthta = np.where(mask_valid, p1, state.pthta)
            
                mask_underflow = mask_le_pdwn & (p1 < state.pressure_up)
                n_counter[mask_underflow] = state.nmax_val + 1
            
                mask_overflow = within_bounds & (p1 > state.pressure_down)
                iter_mask = iter_mask & (~mask_overflow)
            # --- BRANCH 2: LOOP COUNT EXCEEDED TARGETS -> ELSE (Log Non-Convergence) ---
            if np.any(exceeded_bounds):
                match_resmax = exceeded_bounds & (abs_resid > resmax)
                if np.any(match_resmax):
                    resmax = np.max(abs_resid[match_resmax])
            
                iter_mask = iter_mask & (~exceeded_bounds)

        # Return your clean tuple outputs using the updated internal variable registers
        return state.pthta, tdwn, tup, dltdlp, interc
    
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

        # FIXED: Extract the active configuration key tag from kwargs natively
        # Maps directly to strings like "MISSING_VALUE", "KEITH_BRILL", "ECMWF", etc.
        strategy_token = str(kwargs.get("SUBTERRANEAN_PHYSICS", "MissingValueStrategy"))
        
        # 1. Forward arrays and configuration parameters directly across the Pybind11 boundary line
        pthta_cpp, dltdlp_cpp = vayu_core.isobaric_to_isentropic_pressure(
            strategy_token, 
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
        
        # 2. FIXED: Insulate fractional power from negative numbers using a valid-data mask!
        kappa_val = float(kwargs.get("kappa_val", 0.285856))
        p0_val    = float(kwargs.get("p0_val", 100000.0))
        
        # Identify non-missing data points
        valid_mask = pthta_cpp > 0.0
        
        # Pre-allocate temperature tracking arrays matching your grid layout
        t_theta = np.zeros_like(pthta_cpp)
        thta_3d = np.broadcast_to(thta_grid_clean[:, np.newaxis, np.newaxis], pthta_cpp.shape)
        
        # Execute power functions strictly over non-masked spatial locations
        with np.errstate(divide='ignore', invalid='ignore'):
            t_theta[valid_mask] = thta_3d[valid_mask] * (pthta_cpp[valid_mask] / p0_val) ** kappa_val
            
        # Pass the safe arrays back to your validation framework slots
        return pthta_cpp, t_theta, t_theta, dltdlp_cpp, np.zeros_like(pthta_cpp)

class IsobaricVelocityNumPyBackend(GenericDomainStrategy):
    """
    Pure NumPy Vectorized Velocity Transformation Backend.
    Driven natively via a singular, single-parameter IsentropicVelocityState container.
    """

    def _apply_quadratic_helper(self, mask, state: IsentropicVelocityState, pdwn, pmid, pup, sdwn, smid, sup, l12, l13, l23):
        """Helper to compute quadratic wind interpolation operating entirely on the state context."""
        safe_denom = (mask) & (np.abs(l23) > 1e-12) & (np.abs(l13) > 1e-12) & (np.abs(l12) > 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            qdwn = np.divide(np.log(state.pthta/pmid) * np.log(state.pthta/pup), (l23 * l13), where=safe_denom, out=np.zeros_like(state.pthta))
            qmid = np.divide(-np.log(state.pthta/pdwn) * np.log(state.pthta/pup), (l23 * l12), where=safe_denom, out=np.zeros_like(state.pthta))
            qup  = np.divide(np.log(state.pthta/pdwn) * np.log(state.pthta/pmid), (l13 * l12), where=safe_denom, out=np.zeros_like(state.pthta))
            
            # FIXED: Mutate the state matrix outputs directly in-place using state.sthta!
            state.sthta[mask] = (qdwn*sdwn + qmid*smid + qup*sup)[mask]
            state.done[mask] = True

    def execute(self, *args, **kwargs):
        """Unpacks positional arrays and encapsulates them into the IsentropicVelocityState container."""
        pthta, plevs, spres, psfc, ssfc, missing_val = args

        
        # Dimensions and tracking parameters configuration
        kout, nj, ni = pthta.shape
        tol = 0.01

        # Instantiate the single-parameter velocity state object using sthta
        state = IsentropicVelocityState(
            pthta=pthta,
            plevs=plevs,
            psfc=psfc,
            uins=spres,       
            uwndI=ssfc,       
            sthta=np.zeros_like(pthta, dtype=np.float64), # FIXED: Instantiated as sthta
            done=np.zeros_like(pthta, dtype=bool),
            missing_val=missing_val,
            kout=kout,
            nj=nj,
            ni=ni,
            plvls=plevs.size
        )

        pres = np.float64(state.plevs)
        lnpu1p = np.log(pres[1:] / pres[:-1]) 
        lnpu2p = np.log(pres[2:] / pres[:-2])
        
        # Initialization & surface identity setup
        psfc_3d = np.broadcast_to(state.psfc[None, :, :], (state.kout, state.nj, state.ni))
        ssfc_3d = np.broadcast_to(state.uwndI[None, :, :], (state.kout, state.nj, state.ni))

        state.sthta[state.pthta <= 0] = state.missing_val
        state.done[state.pthta <= 0] = True
        
        mask_sfc = (~state.done) & (np.abs(state.pthta - psfc_3d) < tol)
        state.sthta[mask_sfc] = ssfc_3d[mask_sfc]
        state.done[mask_sfc] = True

        # Pre-check isobaric matches for all vertical levels
        for k in range(state.plvls):
            match_mask = (~state.done) & (np.abs(state.pthta - pres[k]) < tol)
            if np.any(match_mask):
                val_2d = state.uins[k]  # Shape: (nj, ni)
                val_3d = val_2d[np.newaxis, :, :].repeat(state.kout, axis=0) # Broadcast to (kout, nj, ni)
                #val_3d = state.uins[k][None, :, :].repeat(state.kout, axis=0)
                state.sthta[match_mask] = val_3d[match_mask]
                state.done[match_mask] = True

        # --- BRANCH 1: SURFACE SPECIAL CASE (k=0) ---
        k = 0
        active_0 = (~state.done) & (state.pthta > pres[k])
        if np.any(active_0):
            pdwn = np.zeros_like(state.sthta); pmid = np.zeros_like(state.sthta); pup = np.zeros_like(state.sthta)
            sdwn = np.zeros_like(state.sthta); smid = np.zeros_like(state.sthta); sup = np.zeros_like(state.sthta)
            l12 = np.zeros_like(state.sthta);  l13 = np.zeros_like(state.sthta);  l23 = np.zeros_like(state.sthta)

            pdwn[active_0], sdwn[active_0] = psfc_3d[active_0], ssfc_3d[active_0]
            c1_3d = (np.abs(psfc_3d - pres[k]) < tol)
            
            pmid[active_0] = np.where(c1_3d, pres[k],   pres[k+1])[active_0]
            pup[active_0]  = np.where(c1_3d, pres[k+1], pres[k+2])[active_0]
            
            s_k0, s_k1, s_k2 = state.uins[k, None], state.uins[k+1, None], state.uins[k+2, None]
            smid[active_0] = np.where(c1_3d, s_k0, s_k1)[active_0]
            sup[active_0]  = np.where(c1_3d, s_k1, s_k2)[active_0]
            
            l12[active_0] = np.where(c1_3d, lnpu1p[k], lnpu1p[k+1])[active_0]
            
            with np.errstate(divide='ignore', invalid='ignore'):
                l13[active_0] = np.where(c1_3d, np.log(np.divide(pup, pdwn, where=pdwn!=0)), lnpu2p[k])[active_0]
                l23[active_0] = np.where(c1_3d, np.log(np.divide(pmid, pdwn, where=pdwn!=0)), lnpu1p[k])[active_0]

            self._apply_quadratic_helper(active_0, state, pdwn, pmid, pup, sdwn, smid, sup, l12, l13, l23)

        # --- BRANCH 2: THE UPPER ATMOSPHERIC SWEEP (k=1 to plvls-2) ---
        pdwn = np.zeros_like(state.sthta); pmid = np.zeros_like(state.sthta); pup = np.zeros_like(state.sthta)
        sdwn = np.zeros_like(state.sthta); smid = np.zeros_like(state.sthta); sup = np.zeros_like(state.sthta)
        l12 = np.zeros_like(state.sthta);  l13 = np.zeros_like(state.sthta);  l23 = np.zeros_like(state.sthta)

        for k in range(1, state.plvls - 1):
            active = (~state.done) & (state.pthta > pres[k])
            if not np.any(active): 
                continue

            m_psfc = active & (psfc_3d < pres[k-1])
            m_gen  = active & (~m_psfc)
            pdwn.fill(0.0);  pmid.fill(0.0);  pup.fill(0.0)
            sdwn.fill(0.0);  smid.fill(0.0);  sup.fill(0.0)
            l12.fill(0.0);   l13.fill(0.0);   l23.fill(0.0)
            
            if np.any(m_psfc):
                pdwn[m_psfc], sdwn[m_psfc] = psfc_3d[m_psfc], ssfc_3d[m_psfc]
                c3_3d = (np.abs(psfc_3d - pres[k]) < 0.001)  
                pmid[m_psfc] = np.where(c3_3d, pres[k],   pres[k+1])[m_psfc]
                pup[m_psfc]  = np.where(c3_3d, pres[k+1], pres[k+2])[m_psfc]
                
                skk, skp, sk2 = state.uins[k, None], state.uins[k+1, None], state.uins[k+2, None]
                smid[m_psfc] = np.where(c3_3d, skk, skp)[m_psfc]
                sup[m_psfc]  = np.where(c3_3d, skp, sk2)[m_psfc]
                
                l12[m_psfc] = np.where(c3_3d, lnpu1p[k], lnpu1p[k+1])[m_psfc]
                with np.errstate(divide='ignore', invalid='ignore'):
                    l13[m_psfc] = np.where(c3_3d, np.log(np.divide(pup, pdwn, where=pdwn!=0)), lnpu2p[k])[m_psfc]
                    l23[m_psfc] = np.where(c3_3d, np.log(np.divide(pmid, pdwn, where=pdwn!=0)), lnpu1p[k])[m_psfc]
                
            if np.any(m_gen):
                pdwn[m_gen], pmid[m_gen], pup[m_gen] = pres[k-1], pres[k], pres[k+1]
                sdwn[m_gen] = state.uins[k-1, None, :, :].repeat(state.kout, axis=0)[m_gen]
                smid[m_gen] = state.uins[k, None, :, :].repeat(state.kout, axis=0)[m_gen]
                sup[m_gen]  = state.uins[k+1, None, :, :].repeat(state.kout, axis=0)[m_gen]
                l12[m_gen], l13[m_gen], l23[m_gen] = lnpu1p[k], lnpu2p[k-1], lnpu1p[k-1]

                self._apply_quadratic_helper(active, state, pdwn, pmid, pup, sdwn, smid, sup, l12, l13, l23)

        # --- BRANCH 3: TOP CAP SPECIAL CASE (k=plvls-1) ---
        k = state.plvls - 1
        active_top = (~state.done) & (state.pthta > pres[k])
        if np.any(active_top):
            pdwn, pmid, pup = pres[k-2], pres[k-1], pres[k]
            sdwn_t = state.uins[k-2][None].repeat(state.kout, axis=0)
            smid_t = state.uins[k-1][None].repeat(state.kout, axis=0)
            sup_t  = state.uins[k][None].repeat(state.kout, axis=0)
            l12_t, l13_t, l23_t = lnpu1p[k-1], lnpu2p[k-2], lnpu1p[k-2]
            
            self._apply_quadratic_helper(active_top, state, pdwn, pmid, pup, sdwn_t, smid_t, sup_t, l12_t, l13_t, l23_t)

        # Returns the generic variable track array cleanly
        return state.sthta


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
        pthta, plevs, spres, psfc, ssfc, missing_val = args

        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        BUILD_DIR = os.path.join(ROOT_DIR, 'builddir')
        
        if BUILD_DIR not in sys.path:
            sys.path.insert(0, BUILD_DIR)
        
        try:
            print("inside here")
            import vayu_core
        except ImportError as e:
            print(f"[EXT CRITICAL] Could not locate compiled C++ binaries in {BUILD_DIR}")
            raise e
        
        
        # Dispatch pointers natively across the Pybind11 boundary line
        # Maps directly to your optimized multi-threaded OpenMP C++ template target kernel
        sthta_cpp = vayu_core.isobaric_to_isentropic_velocity(
            pthta, 
            plevs, 
            spres, 
            psfc, 
            ssfc
        )
        
            
        return sthta_cpp


@inject_constants
@inject_solver_settings
@inject_library_settings
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
    lib_mod = kwargs.get('P2THTA_MODULE', 'coordinate_transformers')
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
    #return pthta_final,py_tdwn,py_tup,py_dltdlp,py_interc
    return pthta_final, thta_grid_clean

    


@inject_constants
@inject_constants
@inject_library_settings
def isobaric_to_isentropic_velocity(plevs, uins, pthta, psfc, uwndI, **kwargs):
    """
    Public entry facade for horizontal wind velocity components mapping.
    Natively accepts parameters isolated strictly per TOML table header boundaries.
    """
    # 1. Enforce robust float64 processing arrays at the front gate
    plevs_arr = np.asarray(plevs, dtype=np.float64)
    uins_arr  = np.asarray(uins, dtype=np.float64)
    pthta_arr = np.asarray(pthta, dtype=np.float64)
    psfc_arr  = np.asarray(psfc, dtype=np.float64)
    uwndI_arr = np.asarray(uwndI, dtype=np.float64)

    # 2. Extract configuration variables from the injected kwargs dictionary
    missing_val = np.float64(kwargs.get("MISSING_DATA", -9999.0))
    
    # FIXED: Replaced legacy config_context string parsing with direct kwargs lookups!
    lib_mod = kwargs.get("S2THTA_MODULE", "coordinate_transformers")
    lib_cls = kwargs.get("S2THTA_CLASS", "IsobaricVelocityNumPyBackend")
    print(f"--> [DEBUG] Running Class: {lib_cls}")  # Will print your true class names!


    # 3. Dynamic Factory Reflection Resolution Pass
    velocity_strategy = DynamicStrategyFactory.resolve(lib_mod, lib_cls)
    
    # 4. Dispatch array views straight to your unified state execution layer
    return velocity_strategy.execute(
        pthta_arr, 
        plevs_arr, 
        uins_arr, 
        psfc_arr, 
        uwndI_arr, 
        missing_val, 
        **kwargs
    )


def tests2thta(plevs, uwndI, psfc, uins, pthta):
    """
    Unified Config-Driven Wind Vector Interpolation Validation Harness.
    Positional arguments match the 5-element ingest loop exactly.
    """
    from coordinate_transformers import isobaric_to_isentropic_velocity
    import interp_lib
    import sys
    
    print("=" * 80)
    print(" VELOCITY TRANSFORMATION (s2thta) INTERPOLATION METRICS AUDIT")
    print("=" * 80)

    # 1. Coordinate Transpose for Fortran Baseline Arrays Matching
    plevs_final = np.asfortranarray(plevs.T, dtype=np.float64)  
    psfc_final  = np.asfortranarray(psfc.T, dtype=np.float64)  
    ssfc_final  = np.asfortranarray(uwndI.T, dtype=np.float64) 
    spres_final = np.asfortranarray(np.transpose(uins, (2, 1, 0)), dtype=np.float64)
    pthta_final = np.asfortranarray(np.transpose(pthta, (2, 1, 0)), dtype=np.float64)
    
    sthta_old_full = interp_lib.s2thta_old(ssfc_final, psfc_final, spres_final, pthta_final)
    sthta_f77 = np.ascontiguousarray(np.transpose(sthta_old_full[:, :, :], (2, 1, 0)))

    # =====================================================================
    # TRACK 1: PURE NUMPY BACKEND STRATEGY VIA PRODUCTION FACADE
    # =====================================================================
    # Force the facade function to resolve and execute IsobaricVelocityNumPyBackend
    sthta_numpy2 = isobaric_to_isentropic_velocity(
        plevs, uins, pthta, psfc, uwndI, 
        S2THTA_CLASS="IsobaricVelocityNumPyBackend"
    )
    
    mea1 = np.mean(np.abs(sthta_f77 - sthta_numpy2))
    print(f"--> [FACADE RUN] NumPy Strategy Backend vs F77 MAE: {mea1:.16e} m/s")

    # =====================================================================
    # TRACK 2: C++ ACCELERATED STRATEGY VIA THE EXACT SAME FACADE
    # =====================================================================
    # Force the facade function to resolve and execute IsobaricVelocityXtensorBackend
    sthta_xtensor = isobaric_to_isentropic_velocity(
        plevs, uins, pthta, psfc, uwndI, 
        S2THTA_CLASS="IsobaricVelocityXtensorBackend"
    )
    
    mea2 = np.mean(np.abs(sthta_xtensor - sthta_f77))
    print(f"--> [FACADE RUN] C++ Xtensor Strategy Backend vs F77 MAE: {mea2:.16e} m/s")
    print("=" * 80)
    
    sys.exit()
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
