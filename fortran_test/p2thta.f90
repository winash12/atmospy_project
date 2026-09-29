SUBROUTINE P2THTA(NI, NJ, LAT, LON, TSFC, PSFC, TPRES, KTHTA,THTA, PTHTA)

  
  INTEGER MAXLVL
  PARAMETER (MAXLVL = 50)
  INTEGER PLVLS
  PARAMETER (PLVLS = 16)
  REAL CP
  PARAMETER (CP = 1004.)
  REAL  MD
  PARAMETER (MD = 28.9644)
  REAL  R
  PARAMETER (R = 8314.41)
  REAL  RD
  PARAMETER (RD = R / MD)
  REAL  KAPPA
  PARAMETER (KAPPA = RD/CP)

  INTEGER I
  INTEGER J
  INTEGER KIN
  INTEGER KOUT
  INTEGER KTHTA
  INTEGER MAXIT
  INTEGER N
  INTEGER NI
  INTEGER NJ
  INTEGER NMAX
  INTEGER NPTS

  REAL ALOGPD
  REAL DFDP
  REAL DLTDLP
  REAL EPSLN
  REAL F
  REAL INTERC
  REAL LAT(*)
  REAL LON(*)
  REAL P1
  REAL PDWN
  REAL POT
  REAL POTDWN
  REAL POTSFC(NIMAX,NJMAX)
  REAL POTUP
  REAL ALOGP(PLVLS)
  REAL ALOGPU
  REAL DTHTA

  REAL PRES (PLVLS)
  REAL PSFC (NIMAX, NJMAX)
  REAL PTHTA (NIMAX, NJMAX, MAXLVL)
  REAL PUP
  REAL RESID
  REAL RESMAX
  REAL T1
  REAL TDWN
  REAL THTA(MAXLVL)
  REAL THTAHI
  REAL THTALO
  REAL THTAP(NIMAX,NJMAX, PLVLS)
  REAL TPRES(NIMAX, NJMAX,*)
  REAL TSFC(NIMAX,NJMAX)
  REAL TUP

  DATA DTHTA /5./
  DATA EPSLN /1./
  DATA NMAX /5/

  DATA PRES/100000., 92500., 85000., 70000., 50000., 40000., &
       30000., 25000., 20000., 15000., 10000., 7000.,5000., 3000., 2000., &
       1000./
  THTALO = POT (TSFC (1, 1), PSFC (1, 1))
  DO  J = 1, NJ
     DO  I = 1, NI
        POTSFC (I, J) = POT (TSFC (I, J), PSFC (I, J))
        IF (POTSFC (I, J) .LT. THTALO) THTALO = POTSFC (I, J)
     end do
  end do
  
  
  THTAHI = POT (TPRES (1, 1, 10), PRES (10))
  DO  KIN = 1, PLVLS
     DO  J = 1, NJ
        DO  I = 1, NI
           THTAP (I, J, KIN) = POT (TPRES (I, J, KIN), PRES (KIN))
           
           IF (PSFC (I, J) .GT. PRES (KIN) ) THEN
              IF (KIN .GT. 1) THEN
                 IF (PSFC(I,J) .LT.PRES(KIN-1)) THEN
                    IF (THTAP(I,J,KIN) .LT.  POTSFC(I,J) ) THEN
                       THTAP(I,J,KIN) = POTSFC(I,J) + 0.01
                    END IF
                 ELSE IF (THTAP(I,J,KIN) .LE. THTAP(I,J,KIN-1)) THEN
                    THTAP(I,J,KIN) = THTAP(I,J,KIN-1) + 0.01
                 END IF
              ELSE
                 IF(THTAP(I,J,1) .LE.POTSFC(I,J)) THEN
                    THTAP(I,J,1) = POTSFC(I,J) + 0.01
                 END IF
              END IF
           END IF
           IF (KIN .GE. 10 .AND. THTAP(I,J,KIN) .GT. THTAHI) THEN
              THTAHI = THTAP(I,J,KIN)
           END IF
        end do
     end do
  end do
  
  KOUT = 0
600 CONTINUE
  KOUT = KOUT + 1
  THTA (1) = 200. + FLOAT (KOUT - 1) * DTHTA
  IF (THTA (1) + DTHTA .GE. THTALO) GO TO 700
  GO TO 600
700 CONTINUE
  THTA (1) = THTA (1) + DTHTA
  NPTS = 0
  J = 0
800 CONTINUE
  J=J+1
  IF (J .LE. NJ) THEN
     I = 0
900  CONTINUE
     I=I+1
     IF (I .LE. NI) THEN
        IF (POTSFC (I, J) .LE. THTA (1)) NPTS = NPTS + 1
        IF (NPTS .GE. FLOAT (NI * NJ) / 10.) GO TO 1000
        GO TO 900
     ELSE
        GO TO 800
     END IF
  ELSE
     GO TO 700
  END IF
1000 CONTINUE
  PRINT *, 'FIRST ISENTROPIC LEVEL IS', THTA (1),'K.'
  KTHTA = 1
1100 CONTINUE
  IF (KTHTA .LE. MAXLVL) THEN
     IF (THTA (KTHTA) .LE. THTAHI) THEN
        THTA (KTHTA + 1) = THTA (KTHTA) + DTHTA
        KTHTA = KTHTA + 1
        GO TO 1100
     END IF
  END IF
1300 CONTINUE
  KTHTA = KTHTA - 1
  NPTS = 0
  J = 0
1400 CONTINUE
  J =J+ 1
  IF (J .LE. NJ) THEN
   I = 0
1500 CONTINUE
   I=I+1
   IF (I .LE. NI) THEN
      IF (THTAP (I, J, PLVLS) .GE. THTA (KTHTA) ) NPTS = NPTS + 1
      IF (FLOAT (NPTS) .GE. FLOAT (NI * NJ) / 10. ) GO TO 1600
      GO TO 1500
   ELSE
      GO TO 1400
   END IF
ELSE
   GO TO 1300
END IF
1600 CONTINUE
IF (KTHTA .GE. MAXLVL) THEN
   PRINT *,'P2THTA: ONLY THE FIRST',MAXLVL,&
        ' ISENTROPIC LEVELS WILL BE CALCULATED. INCREASE ',&
        'MAXLVL PARAMETER OR DTHTA TO OBTAIN DATA ABOVE ',&
        THTA (MAXLVL), 'K.'
ELSE
   PRINT *, 'TOP ISENTROPIC LEVEL ', THTA (KTHTA)
END IF
PRINT *, 'INTERPOLATING PRESSURE TO', KTHTA, ' LEVELS NOW...'

DO  KIN = 1, PLVLS
   ALOGP(KIN) = ALOG(PRES (KIN))
end do
   
MAXIT = 0
RESMAX = 1.
DO   KOUT = 1, KTHTA
   DO   J = 1, NJ
      DO   I = 1, NI
         IF (THTA (KOUT) .LT. POTSFC(I, J)) THEN
            PTHTA (I, J, KOUT) =-9999.
         ELSE IF (THTA (KOUT) .GT. THTAP(I, J, PLVLS) )THEN
            PTHTA (I, J, KOUT) = -9999.
         ELSE IF (ABS (THTA (KOUT) - POTSFC(I, J) ) .LT. 0.001) THEN
            PTHTA (I, J, KOUT) = PSFC (I, J)
         ELSE
            KIN = 0
1700        CONTINUE
            KIN =KIN + 1
            IF (KIN .LE. PLVLS) THEN
               IF (THTA (KOUT) .LT. THTAP (I, J, KIN) )THEN
                  IF (KIN .EQ. 1) THEN
                     POTDWN = POTSFC (I, J)
                     PDWN = PSFC (I, J)
                     ALOGPD = ALOG (PSFC (I, J))
                     IF (ABS(PSFC(I, J) - PRES(KIN)) .LT. 0.001) THEN
                        POTUP = THTAP (I, J, KIN + 1)
                        PUP = PRES (KIN + 1)
                        ALOGPU =  ALOGP (KIN + 1)
                     ELSE
                        POTUP =THTAP(I, J, KIN)
                        PUP = PRES (KIN)
                        ALOGPU =ALOGP (KIN)
                     END IF
                     ELSE IF (POTSFC (I, J) .GT. THTAP (I, J, KIN - 1)) THEN
                        POTDWN = POTSFC (I, J)
                        PDWN = PSFC (I, J)
                        ALOGPD = ALOG (PSFC (I, J))
                        IF (ABS (PSFC (I, J) - PRES (KIN) ) .LT. 0.01) THEN
                           POTUP = TIITAP (I, J, KIN + 1)
                           PUP = PRES (KIN + 1)
                           ALOGPU =ALOGP(KIN + 1)
                        ELSE
                           POTUP =THTAP(I, J, KIN)
                           PUP = PRES (KIN)
                           ALOGPU = ALOGP(KIN)
                        END IF
                     ELSE
                        POTUP = THTAP (I, J, KIN)
                        PUP = PRES (KIN)
                        ALOGPU = ALOGP (KIN)
                        POTDWN = THTAP (I, J, KIN)
                        PDWN = PRES (KIN - 1)
                        ALOGPD = ALOGP (KIN - 1)
                     END IF
                     GO TO 1800
                  ELSE
                     GO TO 1700
                  END IF
               END IF
1800           CONTINUE
               TDWN =POTDWN* (PDWN / 100000.)**KAPPA
               TUP =POTUP* (PUP /100000.)**KAPPA
               DLTDLP = ALOG (TUP /TDWN)/ (ALOGPU - ALOGPD)
               INTERC = ALOG (TUP) -DLTDLP* ALOGPU
               PTHTA (I, J, KOUT) =EXP( (ALOG (THTA (KOUT) )-INTERC-&
                    KAPPA * ALOGP (1))/ &
                    (DLTDLP - KAPPA))
               N = 0
1900           CONTINUE
               T1 = EXP (DLTDLP * ALOG (PTHTA (I, J, KOUT))+ INTERC)
               RESID = PTHTA (I, J, KOUT) -&
                    100000. * (T1 / THTA (KOUT) )**(1. / KAPPA)
               IF (ABS (RESID) .GT. EPSLN) THEN
                  N =N+l1
                  IF (N .LE. NMAX) THEN
                     THTAl = T1 * (100000. / PTI-TA (I, J, KOUT) )**KAPPA
                     F =THTA (KOUT) - THTA1
                     DFDP = (KAPPA - DLTDLP) * &
                          (100000. / PTHTA (I, J, KOUT))**KAPPA *&
                          EXP (INTERC + (DLTDLP - 1.) *&
                          ALOG(PTHTA (I, J, KOUT)))
                     
                     P1 = PTHTA (I, J, KOUT) -F / DFDP
                     IF (P1 .LE. PDWN) THEN
                        IF (P1 .GE. PUP) THEN
                           PTHTA (I, J, KOUT) =P1
                           GO TO 1900
                        ELSE
                           N = NMAX
                        END IF
                     END IF
                  ELSE
                     IF (RESID .GT. RESMAX) RESMAX = RESID
                     MAXIT = MAXIT + 1
                     GO TO 2100
                  END IF
               END IF
2100           CONTINUE
               if (PTHTA (I, J, KOUT - 1) .GT. 0.) THEN
                  IF (PTHTA (I, J, KOUT) .GT. PTHTA (I, J, KOUT - 1)) then
                     PTHTA (I, J, KOUT) = PTHTA (I, J, KOUT - 1) + 0.001
               END IF
            END IF
         END IF
      END DO
      NPTS = 0
      IF (ABS (LAT (J)) .GE. 90.) THEN
         IF (PTHTA (1, J, KOUT) .GT. 0.) NPTS = 1
         DO  I = 2, NI - 1
            IF (PTHTA (1, J, KOUT) .GT. 0.) THEN
               IF (PTHTA (I, J, KOUT) .GT. 0.) THEN
                  PTHTA (1, J, KOUT) = PTHTA (1, J, KOUT) +&
                       PTHTA (I, J, KOUT)
                  NPTS = NPTS +1                    
               END IF
            ELSE
               IF (PTHTA (I, J, KOUT) .GT. 0.) THEN
                  PTHTA (1, J, KOUT) = PTHTA (I, J, KOUT)
                  NPTS = NPTS + 1
               END IF
            END IF
         END DO
         IF (NPTS .EQ. 0) GOTO 2500
         IF (ABS(LON(1)-LON(NI)) .LT. 0.001) THEN
            PTHTA(1,J,KOUT) = PTHTA(1,J,KOUT)/FLOAT(NPTS)
         ELSE IF (PTHTA(NI,J,KOUT) .GT. 0) THEN
            PTHTA(1,J,KOUT) = PTHTA(1,J,KOUT) + PTHTA(NI,J,KOUT)
            PTHTA(1,J,KOUT) = PTHTA(1,J,KOUT)/FLOAT(NPTS+1)
         ELSE
            PTHTA(1,J,KOUT) = PTHTA(1,J,KOUT)/FLOAT(NPTS)
         END IF
         DO  I=2,NI
            PTHTA(I,J,NOUT) = PTHTA(1,J,KOUT)
         END DO
      END IF
   END DO
END DO
return
end subroutine p2thta
             
