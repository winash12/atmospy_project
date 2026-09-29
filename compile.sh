python3.12 -m numpy.f2py -c s2thta_f23.f90 -m weather_lib --f90flags="-O3 -march=native"
