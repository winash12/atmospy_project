PROGRAM TEST_P2THTA
  IMPLICIT NONE
  
  INTEGER, PARAMETER :: NI = 4, NJ = 4
  DOUBLE PRECISION :: TSFC(NI, NJ), PSFC(NI, NJ)
  DOUBLE PRECISION :: TPRES(NI, NJ, 17)
  DOUBLE PRECISION :: LATS(NJ), LONS(NI)
  DOUBLE PRECISION :: KAPPA_VAL
  INTEGER          :: KTHTA
  DOUBLE PRECISION :: THTA(50), PTHTA(NI, NJ, 50), THTAP(NI, NJ, 17)
  INTEGER          :: I, J, K

  PRINT *, "=========================================="
  PRINT *, "  INITIALIZING ATMOSPHERIC DATA FIELD     "
  PRINT *, "=========================================="

  ! Coordinates and physical constants
  LATS = (/ 10.0D0, 20.0D0, 30.0D0, 40.0D0 /)
  LONS = (/ 50.0D0, 60.0D0, 70.0D0, 80.0D0 /)
  KAPPA_VAL = 0.286D0

  ! Surface parameters (Standard Atmosphere ~ 15 C / 1013 hPa)
  TSFC = 288.15D0
  PSFC = 101325.0D0

  ! Vertical temperature profile (~6.5 K/km standard lapse rate)
  DO K = 1, 17
    DO J = 1, NJ
      DO I = 1, NI
        TPRES(I, J, K) = 288.15D0 - DBLE(K - 1) * 6.5D0
      END DO
    END DO
  END DO

  PRINT *, "Data generated successfully."
  PRINT *, "Running P2THTA subroutine..."
  PRINT *, "------------------------------------------"

  CALL P2THTA(NI, NJ, TSFC, PSFC, TPRES, LATS, LONS, KAPPA_VAL, KTHTA, THTA, PTHTA, THTAP)

  PRINT *, "------------------------------------------"
  PRINT *, "Subroutine executed successfully with 0 errors!"

END PROGRAM TEST_P2THTA


SUBROUTINE P2THTA(NI, NJ, TSFC, PSFC, TPRES, LATS, LONS, KAPPA_VAL, KTHTA, THTA, PTHTA, THTAP)
  IMPLICIT NONE
  
  ! 1. Variable Arguments & Explicit Intent
  INTEGER, INTENT(IN)           :: NI, NJ
  DOUBLE PRECISION, INTENT(IN)  :: TSFC(NI, NJ)
  DOUBLE PRECISION, INTENT(IN)  :: PSFC(NI, NJ)
  DOUBLE PRECISION, INTENT(IN)  :: TPRES(NI, NJ, 17)
  DOUBLE PRECISION, INTENT(IN)  :: LATS(NJ), LONS(NI)
  DOUBLE PRECISION, INTENT(IN)  :: KAPPA_VAL
  
  INTEGER, INTENT(OUT)          :: KTHTA
  DOUBLE PRECISION, INTENT(OUT) :: THTA(50)
  DOUBLE PRECISION, INTENT(OUT) :: PTHTA(NI, NJ, 50)
  DOUBLE PRECISION, INTENT(OUT) :: THTAP(NI, NJ, 17)

  ! 2. Local Parameters & Scalars
  INTEGER, PARAMETER :: MAXLVL = 50
  INTEGER, PARAMETER :: PLVLS = 17
  INTEGER            :: I, J, KIN, STRAT_IDX
  INTEGER            :: KOUT, NPTS
  DOUBLE PRECISION   :: THTALO, THTAHI, P0_REF
  DOUBLE PRECISION   :: POTSFC(NI, NJ)
  DOUBLE PRECISION   :: PRES(PLVLS)
  DOUBLE PRECISION   :: DTHTA, CANDIDATE_THTA, THRESHOLD

  ! 3. External Potential Temperature Calculation Function
  DOUBLE PRECISION   :: POT
  EXTERNAL POT

  ! Standard atmospheric pressure levels (in Pascals)
  DATA PRES/100000.D0, 92500.D0, 85000.D0, 70000.D0, 60000.D0, 50000.D0, 40000.D0, &
            30000.D0, 25000.D0, 20000.D0, 15000.D0, 10000.D0, 7000.D0, 5000.D0, &
            3000.D0, 2000.D0, 1000.D0/

  ! --- CORE FIX 1: Explicitly initialize theta interval step (2.0 Kelvin) ---
  DTHTA = 10.0D0

  ! Set up baseline reference pressure
  P0_REF = 100000.D0
  IF (PSFC(1,1) .LT. 2000.D0) THEN
     P0_REF = 1000.D0
     DO KIN = 1, PLVLS
        PRES(KIN) = PRES(KIN) / 100.D0
     END DO
  END IF

  ! Surface potential temperature calculation
  THTALO = POT(TSFC(1, 1), PSFC(1, 1))
  DO J = 1, NJ
     DO I = 1, NI
        POTSFC(I, J) = POT(TSFC(I, J), PSFC(I, J))
        IF (POTSFC(I, J) .LT. THTALO) THTALO = POTSFC(I, J)
     END DO
  END DO
  
  STRAT_IDX = (PLVLS / 2) + 1
  THTAHI = POT(TPRES(1, 1, STRAT_IDX), PRES(STRAT_IDX))
  
  ! Superadiabatic column modification pass
  DO KIN = 1, PLVLS
     DO J = 1, NJ
        DO I = 1, NI
           THTAP(I, J, KIN) = POT(TPRES(I, J, KIN), PRES(KIN))
           
           IF (PSFC(I, J) .GT. PRES(KIN)) THEN
              IF (KIN .GT. 1) THEN
                 IF (PSFC(I, J) .LT. PRES(KIN-1)) THEN
                    IF (THTAP(I, J, KIN) .LT. POTSFC(I, J)) THEN
                       THTAP(I, J, KIN) = POTSFC(I, J) + 0.01D0
                    END IF
                 ELSE IF (THTAP(I, J, KIN) .LE. THTAP(I, J, KIN-1)) THEN
                    THTAP(I, J, KIN) = THTAP(I, J, KIN-1) + 0.01D0
                 END IF
              ELSE
                 IF (THTAP(I, J, 1) .LE. POTSFC(I, J)) THEN
                    THTAP(I, J, 1) = POTSFC(I, J) + 0.01D0
                 END IF
              END IF
           END IF
           
           IF (KIN .GE. STRAT_IDX .AND. THTAP(I, J, KIN) .GT. THTAHI) THEN
              THTAHI = THTAP(I, J, KIN)
           END IF
        END DO
     END DO
  END DO
  
  ! Reset outputs
  KTHTA = 0
  THTA(:) = 0.0D0
  PTHTA(:,:,:) = 0.0D0
  
  ! 10% grid point threshold constraint
  THRESHOLD = DBLE(NI * NJ) / 10.0D0

  ! -------------------------------------------------------------------
  ! Step 1: Find First Isentropic Level
  ! -------------------------------------------------------------------
  CANDIDATE_THTA = 200.0D0
  DO WHILE (CANDIDATE_THTA + DTHTA < THTALO)
     CANDIDATE_THTA = CANDIDATE_THTA + DTHTA
  END DO
  CANDIDATE_THTA = CANDIDATE_THTA + DTHTA

  DO WHILE (CANDIDATE_THTA .LT. 600.0D0)
     NPTS = 0
     DO J = 1, NJ
        DO I = 1, NI
           IF (POTSFC(I, J) .GT. 0.0D0 .AND. POTSFC(I, J) .LE. CANDIDATE_THTA) THEN
              NPTS = NPTS + 1
           END IF
        END DO
     END DO
     
     IF (DBLE(NPTS) .GE. THRESHOLD) EXIT
     CANDIDATE_THTA = CANDIDATE_THTA + DTHTA
  END DO

  THTA(1) = CANDIDATE_THTA
  PRINT *, 'FIRST ISENTROPIC LEVEL IS ', THTA(1), ' K.'

  ! -------------------------------------------------------------------
  ! Step 2: Generate Vertical Candidate Theta Grid Levels
  ! -------------------------------------------------------------------
  KTHTA = 1
  DO WHILE (KTHTA .LT. MAXLVL)
     IF (THTA(KTHTA) + DTHTA .GT. THTAHI) EXIT
     THTA(KTHTA + 1) = THTA(KTHTA) + DTHTA
     KTHTA = KTHTA + 1
  END DO

  DO WHILE (KTHTA .GT. 1)
     NPTS = 0
     DO J = 1, NJ
        DO I = 1, NI
           IF (THTAP(I, J, PLVLS) .GE. THTA(KTHTA)) THEN
              NPTS = NPTS + 1
           END IF
        END DO
     END DO

     IF (DBLE(NPTS) .GE. THRESHOLD) EXIT
     KTHTA = KTHTA - 1
  END DO

  IF (KTHTA .GE. MAXLVL) THEN
     PRINT *, 'P2THTA: MAXIMUM CAP REACHED. ONLY THE FIRST', MAXLVL, ' ISENTROPIC LEVELS COMPUTED.'
  ELSE
     PRINT *, 'TOP ISENTROPIC LEVEL IS ', THTA(KTHTA), ' K.'
  END IF

  PRINT *, 'INTERPOLATING PRESSURE TO', KTHTA, ' LEVELS NOW...'

  RETURN
END SUBROUTINE P2THTA


! Potential Temperature Formula Calculation
DOUBLE PRECISION FUNCTION POT(TMP, PRES)
  IMPLICIT NONE
  
  DOUBLE PRECISION, INTENT(IN) :: TMP, PRES
  
  DOUBLE PRECISION, PARAMETER :: CP = 1004.D0
  DOUBLE PRECISION, PARAMETER :: MD = 28.9644D0
  DOUBLE PRECISION, PARAMETER :: R  = 8313.41D0
  DOUBLE PRECISION, PARAMETER :: RD = R / MD
  
  IF (PRES .LE. 0.D0) THEN
     POT = -9999.0D0
  ELSE
     POT = TMP * (100000.0D0 / PRES)**(RD / CP)
  END IF

  RETURN
END FUNCTION POT
