PROGRAM MAIN_TEST
  IMPLICIT NONE

  ! 1. Interfaces to match your subroutine signature
  INTERFACE
     SUBROUTINE P2THTA(NI, NJ, PLVLS, MAXLVL, TSFC, PSFC, TPRES, KAPPA_VAL, KTHTA, THTA, PTHTA, THTAP)
       INTEGER, INTENT(IN)           :: NI, NJ, PLVLS, MAXLVL
       DOUBLE PRECISION, INTENT(IN)  :: TSFC(NI, NJ)
       DOUBLE PRECISION, INTENT(IN)  :: PSFC(NI, NJ)
       DOUBLE PRECISION, INTENT(IN)  :: TPRES(NI, NJ, PLVLS)
       DOUBLE PRECISION, INTENT(IN)  :: KAPPA_VAL
       INTEGER, INTENT(OUT)          :: KTHTA
       DOUBLE PRECISION, INTENT(OUT) :: THTA(MAXLVL)
       DOUBLE PRECISION, INTENT(OUT) :: PTHTA(NI, NJ, MAXLVL)
       DOUBLE PRECISION, INTENT(OUT) :: THTAP(NI, NJ, PLVLS)
     END SUBROUTINE P2THTA
  END INTERFACE

  ! 2. Dimension Parameters
  INTEGER, PARAMETER :: NI = 1, NJ = 1, PLVLS = 17, MAXLVL = 50
  
  ! 3. Argument Array Declarations (Double Precision)
  DOUBLE PRECISION :: TSFC(NI, NJ)
  DOUBLE PRECISION :: PSFC(NI, NJ)
  DOUBLE PRECISION :: TPRES(NI, NJ, PLVLS)
  DOUBLE PRECISION :: KAPPA_VAL
  
  INTEGER          :: KTHTA
  DOUBLE PRECISION :: THTA(MAXLVL)
  DOUBLE PRECISION :: PTHTA(NI, NJ, MAXLVL)
  DOUBLE PRECISION :: THTAP(NI, NJ, PLVLS)

  ! Local tracking loop indices
  INTEGER :: K, L

  ! 4. Populate Dummy Atmospheric Data Profile (Single Column)
  KAPPA_VAL = 0.2857142857142857D0
  
  ! Ground baseline observations
  TSFC(1, 1) = 295.15D0  ! Surface Temperature in Kelvin (22 C)
  PSFC(1, 1) = 101325.D0 ! Surface Pressure in Pa (1 atm)

  ! Standard atmospheric temperature profile values (17 Mandatory Levels)
  ! Level 1 (1000 hPa) down to Level 17 (10 hPa)
  TPRES(1, 1, 1)  = 294.0D0  ! 100000 Pa
  TPRES(1, 1, 2)  = 289.0D0  ! 92500 Pa
  TPRES(1, 1, 3)  = 284.0D0  ! 85000 Pa
  TPRES(1, 1, 4)  = 274.0D0  ! 70000 Pa
  TPRES(1, 1, 5)  = 265.0D0  ! 60000 Pa
  TPRES(1, 1, 6)  = 255.0D0  ! 50000 Pa
  TPRES(1, 1, 7)  = 243.0D0  ! 40000 Pa
  TPRES(1, 1, 8)  = 229.0D0  ! 30000 Pa
  TPRES(1, 1, 9)  = 221.0D0  ! 25000 Pa
  TPRES(1, 1, 10) = 216.0D0  ! 20000 Pa
  TPRES(1, 1, 11) = 216.0D0  ! 15000 Pa
  TPRES(1, 1, 12) = 216.0D0  ! 10000 Pa
  TPRES(1, 1, 13) = 218.0D0  ! 7000 Pa
  TPRES(1, 1, 14) = 220.0D0  ! 5000 Pa
  TPRES(1, 1, 15) = 224.0D0  ! 3000 Pa
  TPRES(1, 1, 16) = 227.0D0  ! 2000 Pa
  TPRES(1, 1, 17) = 232.0D0  ! 1000 Pa

  PRINT *, '==================================================='
  PRINT *, 'LAUNCHING SUBROUTINE STANDALONE DRIVER VALIDATION'
  PRINT *, '==================================================='

  ! 5. Invoke your unaltered Fortran subroutine module execution path
  CALL P2THTA(NI, NJ, PLVLS, MAXLVL, TSFC, PSFC, TPRES, KAPPA_VAL, KTHTA, THTA, PTHTA, THTAP)

  ! 6. Stream finalized outputs to terminal console lines
  PRINT *, '---------------------------------------------------'
  PRINT *, 'SUBROUTINE RETURNED SUCCESSFULLY'
  PRINT *, '-> Total Generated Isentropic Levels (KTHTA):', KTHTA
  PRINT *, '---------------------------------------------------'
  
  PRINT *, 'PRINTING SOLVED PRESSURE LEVELS FOR CODES TRACKING:'
  DO K = 1, KTHTA
     PRINT *, 'Level [', K, '] Target Theta =', THTA(K), ' K -> Pressure =', PTHTA(1, 1, K), ' Pa'
  END DO
  PRINT *, '==================================================='

END PROGRAM MAIN_TEST
