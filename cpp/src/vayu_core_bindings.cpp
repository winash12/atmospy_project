#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include <pybind11/pybind11.h>
#include "include/vayu_core_ops.hpp"
#include "include/subterranean_strategies.hpp" 
namespace py = pybind11;

// Wrapper for Horizontal Velocities Track
xt::pyarray<double> isobaric_to_isentropic_velocity_wrapper(
    xt::pyarray<double> pthta, 
    xt::pyarray<double> plevs, 
    xt::pyarray<double> spres, 
    xt::pyarray<double> psfc, 
    xt::pyarray<double> ssfc) {
    
    // 1. Convert views into internal xarray tracking representations 
    // while the GIL is still held by the calling parent thread.
    xt::xarray<double> pthta_cpp = pthta;
    xt::xarray<double> plevs_cpp = plevs;
    xt::xarray<double> spres_cpp = spres;
    xt::xarray<double> psfc_cpp  = psfc;
    xt::xarray<double> ssfc_cpp  = ssfc;
    
    xt::xarray<double> sthta = xt::empty<double>(pthta.shape());
    
    {
        // 2. GIL RELEASE MUTEX BLOCK: Release lock safely now that 
        // underlying raw C++ tensors are completely decoupled from Python heap bounds!
        py::gil_scoped_release release;
        
        // Execute multi-threaded OpenMP kernel targets cleanly
        s2thta_kernel(pthta_cpp, plevs_cpp, spres_cpp, psfc_cpp, ssfc_cpp, sthta);
    }
    
    return sthta;
}

py::tuple isobaric_to_isentropic_pressure_wrapper(
                                                  std::string strategy_name,           // <-- Passed directly from Python's toml extractor!
                                                  xt::pyarray<double> thta_grid,
                                                  xt::pyarray<double> plevs,
                                                  xt::pyarray<double> potsfc,
                                                  xt::pyarray<double> psfc,
                                                  xt::pyarray<double> thtap_cleaned,
                                                  double kappa, double epsln, int nmax, double p0_val, double missing_val) {
  
  // 1. Extract pure C++ array tracking structures while holding GIL
  xt::xarray<double> thta_grid_cpp     = thta_grid;
  xt::xarray<double> plevs_cpp         = plevs;
  xt::xarray<double> potsfc_cpp        = potsfc;
  xt::xarray<double> psfc_cpp          = psfc;
  xt::xarray<double> thtap_cleaned_cpp = thtap_cleaned;
  
  // Updated to match your multi-element computational pair output structures
  std::pair<xt::xarray<double>, xt::xarray<double>> result;
  
    {
      // 2. GIL RELEASE MUTEX BLOCK: Drop lock safely for fused computation loops
      py::gil_scoped_release release;
        
      // 3. Resolve the compile-time template classes exactly ONCE at the gateway entry point!
      if (strategy_name == "KeithBrillStrategy") {
        pv_core::KeithBrillStrategy strategy;
        result = execute_p2thta_fused_core(
                                           strategy, thta_grid_cpp, plevs_cpp, potsfc_cpp, psfc_cpp, thtap_cleaned_cpp, 
                                           kappa, epsln, nmax, p0_val, missing_val
                                           );
      } 
      else if (strategy_name == "ECMWFOrszagStrategy") {
        pv_core::ECMWFOrszagStrategy strategy;
        result = execute_p2thta_fused_core(
                                           strategy, thta_grid_cpp, plevs_cpp, potsfc_cpp, psfc_cpp, thtap_cleaned_cpp, 
                                           kappa, epsln, nmax, p0_val, missing_val
                                           );
      }
      else if (strategy_name == "MissingValueStrategy") {
            pv_core::MissingValueStrategy strategy;
            result = execute_p2thta_fused_core(
                strategy, thta_grid_cpp, plevs_cpp, potsfc_cpp, psfc_cpp, thtap_cleaned_cpp, 
                kappa, epsln, nmax, p0_val, missing_val
            );
      }
      else if (strategy_name == "StrictLorenzStrategy") {
        pv_core::StrictLorenzStrategy strategy;
        result = execute_p2thta_fused_core(
                                           strategy, thta_grid_cpp, plevs_cpp, potsfc_cpp, psfc_cpp, thtap_cleaned_cpp, 
                                           kappa, epsln, nmax, p0_val, missing_val
                                           );
      }
        else {
          throw std::runtime_error("Dynamic Execution Exception: Subterranean strategy type '" + 
                                   strategy_name + "' is unknown in vayu_core context.");
        }
    }
    
    // Return to Python with runtime results packed safely
    return py::make_tuple(result.first, result.second);
}

PYBIND11_MODULE(vayu_core, m) {
  // Initialize standard C-API numpy array tracking tables
  xt::import_numpy();
  
  // Modern JOSS Descriptive Bindings 
  m.def("isobaric_to_isentropic_velocity", &isobaric_to_isentropic_velocity_wrapper, 
        "Accelerated horizontal wind vector isobaric-to-isentropic transformation");
  
  m.def("isobaric_to_isentropic_pressure", &isobaric_to_isentropic_pressure_wrapper,
        "Accelerated loop-fused coordinate tracker engine with 1e-16 target stability");
}

