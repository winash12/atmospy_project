SUBROUTINE P2THTA(NI, NJ, PLVLS, MAXLVL, TSFC, PSFC, TPRES, KAPPA_VAL, KTHTA, THTA, PTHTA, THTAP, &
                  DBG_POTDWN, DBG_PDWN, DBG_POTUP, DBG_PUP)
  IMPLICIT NONE
  
  ! 1. Dimension variables (f2py hides these, calculated from array shapes)
  INTEGER, INTENT(IN)           :: NI, NJ, PLVLS
  
  ! 2. Configuration scalar inputs
  INTEGER, INTENT(IN)           :: MAXLVL
  DOUBLE PRECISION, INTENT(IN)  :: KAPPA_VAL
  
  ! 3. Input Data Grids (Removed LATS and LONS)
  DOUBLE PRECISION, INTENT(IN)  :: TSFC(NI, NJ)
  DOUBLE PRECISION, INTENT(IN)  :: PSFC(NI, NJ)
  DOUBLE PRECISION, INTENT(IN)  :: TPRES(NI, NJ, PLVLS)
  
  ! 4. Output Variables
  INTEGER, INTENT(OUT)          :: KTHTA
  DOUBLE PRECISION, INTENT(OUT) :: THTA(MAXLVL)
  DOUBLE PRECISION, INTENT(OUT) :: PTHTA(NI, NJ, MAXLVL)
  DOUBLE PRECISION, INTENT(OUT) :: THTAP(NI, NJ, PLVLS)

  ! >>> ADD THESE TO INTENT(OUT) FOR THE AUDIT PASS <<<
  DOUBLE PRECISION, INTENT(OUT) :: DBG_POTDWN(NI, NJ, MAXLVL)
  DOUBLE PRECISION, INTENT(OUT) :: DBG_PDWN(NI, NJ, MAXLVL)
  DOUBLE PRECISION, INTENT(OUT) :: DBG_POTUP(NI, NJ, MAXLVL)
  DOUBLE PRECISION, INTENT(OUT) :: DBG_PUP(NI, NJ, MAXLVL)

  DOUBLE PRECISION, PARAMETER :: CP = 1004.D0
  DOUBLE PRECISION, PARAMETER :: MD = 28.9644D0
  DOUBLE PRECISION, PARAMETER :: R  = 8314.41D0
  DOUBLE PRECISION, PARAMETER :: RD = R / MD
  DOUBLE PRECISION, PARAMETER :: KAPPA = RD / CP
  ! 5. Local Declarations
  INTEGER            :: I, J, KIN, STRAT_IDX
  INTEGER            :: KOUT, NPTS,MAXIT,N,NMAX
  DOUBLE PRECISION   :: THTALO, THTAHI, P0_REF
  DOUBLE PRECISION   :: POTSFC(NI, NJ)
  DOUBLE PRECISION   :: PRES(PLVLS)
  DOUBLE PRECISION   :: DTHTA, CANDIDATE_THTA, THRESHOLD
  DOUBLE PRECISION   :: ALOGPD, DFDP,DLTDLP
  DOUBLE PRECISION   :: ALOGPU, EPSLN,F,INTERC
  DOUBLE PRECISION   :: POTDWN,POTUP,PDWN,PUP
  DOUBLE PRECISION   :: RESID,P1
  DOUBLE PRECISION   :: RESMAX, T1, TUP,TDWN,THTA1
  DOUBLE PRECISION, DIMENSION(PLVLS) :: ALOGP
  DOUBLE PRECISION   :: POT
  EXTERNAL POT

  ! --- Executable Code ---
  PRES = (/ 100000.D0, 92500.D0, 85000.D0, 70000.D0, 60000.D0, 50000.D0, 40000.D0, &
             30000.D0, 25000.D0, 20000.D0, 15000.D0, 10000.D0,  7000.D0,  5000.D0, &
              3000.D0,  2000.D0,  1000.D0 /)

 
  EPSLN = 1.0D0
  NMAX  = 5
  THTALO = POT(TSFC(1, 1), PSFC(1, 1))
  DO J = 1, NJ
     DO I = 1, NI
        POTSFC(I, J) = POT(TSFC(I, J), PSFC(I, J))
        IF (POTSFC(I, J) .LT. THTALO) THTALO = POTSFC(I, J)
     END DO
  END DO
  
  STRAT_IDX = (PLVLS / 2) + 1
  THTAHI = POT(TPRES(1, 1, STRAT_IDX), PRES(STRAT_IDX))
  
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
  
  KTHTA = 0
  THTA(:) = 0.0D0
  PTHTA(:,:,:) = 0.0D0
  DTHTA = 5.0D0

  THRESHOLD = DBLE(NI * NJ) / 10.0D0

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
  PRINT *, 'FIRST ISENTROPIC LEVEL IS', THTA(1), 'K.'

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
     PRINT *, 'P2THTA: ONLY THE FIRST', MAXLVL, ' ISENTROPIC LEVELS WILL BE CALCULATED.'
  ELSE
     PRINT *, 'TOP ISENTROPIC LEVEL ', THTA(KTHTA), 'K.'
  END IF

  PRINT *, 'INTERPOLATING PRESSURE TO', KTHTA, ' LEVELS NOW...'
DO  KIN = 1, PLVLS
   ALOGP(KIN) = LOG(PRES (KIN))
end do
   
MAXIT = 0
RESMAX = 1.
 DO KOUT = 1, KTHTA
     DO J = 1, NJ
        DO I = 1, NI
           IF (THTA(KOUT) .LT. POTSFC(I, J)) THEN
              PTHTA(I, J, KOUT) = -9999.D0
           ELSE IF (THTA(KOUT) .GT. THTAP(I, J, PLVLS)) THEN
              PTHTA(I, J, KOUT) = -9999.D0
           ELSE IF (ABS(THTA(KOUT) - POTSFC(I, J)) .LT. 0.001D0) THEN
              PTHTA(I, J, KOUT) = PSFC(I, J)
           ELSE
              KIN = 0
1700          CONTINUE
              KIN = KIN + 1
              IF (KIN .LE. PLVLS) THEN
                 IF (THTA(KOUT) .LT. THTAP(I, J, KIN)) THEN
                    IF (KIN .EQ. 1) THEN
                       POTDWN = POTSFC(I, J)
                       DBG_POTDWN(I,J,KOUT) = POTDWN
                       PDWN = PSFC(I, J)
                       DBG_PDWN(I,J,KOUT) = PDWN
                       ALOGPD = LOG(PSFC(I, J))
                       IF (ABS(PSFC(I, J) - PRES(KIN)) .LT. 0.001D0) THEN
                          POTUP = THTAP(I, J, KIN + 1)
                          DBG_POTUP(I,J,KOUT) = POTUP
                          PUP = PRES(KIN + 1)
                          DBG_PUP(I,J,KOUT) = PUP
                          ALOGPU = ALOGP(KIN + 1)
                       ELSE
                          POTUP = THTAP(I, J, KIN)
                          DBG_POTUP(I,J,KOUT) = POTUP
                          PUP = PRES(KIN)
                          DBG_PUP(I,J,KOUT) = PUP
                          ALOGPU = ALOGP(KIN)
                       END IF
                    ELSE IF (POTSFC(I, J) .GT. THTAP(I, J, KIN - 1)) THEN
                       POTDWN = POTSFC(I, J)
                       DBG_POTDWN(I,J,KOUT) = POTDWN
                       PDWN = PSFC(I, J)
                       DBG_PDWN(I,J,KOUT) = PDWN
                       ALOGPD = LOG(PSFC(I, J))
                       IF (ABS(PSFC(I, J) - PRES(KIN)) .LT. 0.01D0) THEN

                          POTUP = THTAP(I, J, KIN + 1)
                          DBG_POTUP(I,J,KOUT) = POTUP
                          PUP = PRES(KIN + 1)
                          DBG_PUP(I,J,KOUT) = PUP
                          ALOGPU = ALOGP(KIN + 1)
                       ELSE
                          POTUP = THTAP(I, J, KIN)
                          DBG_POTUP(I,J,KOUT) = POTUP
                          PUP = PRES(KIN)
                          DBG_PUP(I,J,KOUT) = PUP
                          ALOGPU = ALOGP(KIN)
                       END IF
                    ELSE
                       POTUP = THTAP(I, J, KIN)
                       DBG_POTUP(I,J,KOUT) = POTUP
                       PUP = PRES(KIN)
                       DBG_PUP(I,J,KOUT) = PUP
                       ALOGPU = ALOGP(KIN)
                       POTDWN = THTAP(I, J, KIN - 1)
                       DBG_POTDWN(I,J,KOUT) = POTDWN
                       PDWN = PRES(KIN - 1)
                       DBG_PDWN(I,J,KOUT) = PDWN
                       ALOGPD = ALOGP(KIN - 1)
                    END IF
                    GO TO 1800
                 ELSE
                    GO TO 1700
                 END IF
              END IF
1800          CONTINUE
              
              TDWN = POTDWN * (PDWN / 100000.D0)**KAPPA
              TUP = POTUP * (PUP / 100000.D0)**KAPPA
              DLTDLP = LOG(TUP / TDWN) / (ALOGPU - ALOGPD)
              INTERC = LOG(TUP) - DLTDLP * ALOGPU
              PTHTA(I, J, KOUT) = EXP((LOG(THTA(KOUT)) - INTERC - KAPPA * ALOGP(1)) / (DLTDLP - KAPPA))
              
              N = 0
1900          CONTINUE
              T1 = EXP(DLTDLP * LOG(PTHTA(I, J, KOUT)) + INTERC)
              RESID = PTHTA(I, J, KOUT) - 100000.D0 * (T1 / THTA(KOUT))**(1.D0 / KAPPA)
              
              IF (ABS(RESID) .GT. EPSLN) THEN
                 N = N + 1
                 IF (N .LE. NMAX) THEN
                    THTA1 = T1 * (100000.D0 / PTHTA(I, J, KOUT))**KAPPA
                    F = THTA(KOUT) - THTA1
                    DFDP = (KAPPA - DLTDLP) * (100000.D0 / PTHTA(I, J, KOUT))**KAPPA * &
                           EXP(INTERC + (DLTDLP - 1.D0) * LOG(PTHTA(I, J, KOUT)))
                    
                    P1 = PTHTA(I, J, KOUT) - F / DFDP
                    IF (P1 .LE. PDWN) THEN
                       IF (P1 .GE. PUP) THEN
                          PTHTA(I, J, KOUT) = P1
                          GO TO 1900
                       ELSE
                          N = NMAX
                       END IF
                    END IF
                 ELSE
                    IF (ABS(RESID) .GT. RESMAX) RESMAX = ABS(RESID)
                    MAXIT = MAXIT + 1
                    GO TO 2100
                 END IF
              END IF
           END IF
2100       CONTINUE
           IF (KOUT .GT. 1) THEN
              IF (PTHTA(I,J,KOUT-1) .GT. 0.0D0) THEN
                 IF (PTHTA(I,J,KOUT) .GT. PTHTA(I,J,KOUT-1)) THEN
                    PTHTA(I,J,KOUT) = PTHTA(I,J,KOUT-1) + 0.001D0
                 END IF
              END IF
           END IF
        END DO
     END DO
  END DO
  
  RETURN
END SUBROUTINE P2THTA

DOUBLE PRECISION FUNCTION POT(TMP, PRES)
  IMPLICIT NONE
  DOUBLE PRECISION, INTENT(IN) :: TMP, PRES
  DOUBLE PRECISION, PARAMETER  :: CP = 1004.D0
  DOUBLE PRECISION, PARAMETER  :: MD = 28.9644D0
  DOUBLE PRECISION, PARAMETER  :: R  = 8314.41D0
  DOUBLE PRECISION, PARAMETER  :: RD = R / MD
  DOUBLE PRECISION, PARAMETER  :: KAPPA_BASE = RD / CP
  
  IF (PRES .LE. 0.D0) THEN
     POT = -9999.0D0
  ELSE
     ! FIXED: Append .D0 to your constants to force 64-bit precision tracking
     POT = TMP * (100000.D0 / PRES)**KAPPA_BASE
  END IF

  RETURN
END FUNCTION POT
