SUBROUTINE S2THTA_OLD(NI, NJ, KIN, KOUT, SSFC, PSFC, SPRES, PTHTA, STHTA)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NI, NJ, KIN, KOUT
  DOUBLE PRECISION, INTENT(IN)  :: SPRES(NI, NJ, KIN)   ! This is 17
  DOUBLE PRECISION, INTENT(IN)  :: PTHTA(NI, NJ, KOUT)  ! This is 16
  ! 1. The variables passed in as arguments (Do NOT declare these as local INTEGERS again)
  ! 2. Actual Local Integers for loops
  INTEGER, PARAMETER :: MAXLVL = 50
  INTEGER, PARAMETER :: PLVLS = 17
  INTEGER :: I, J,K_IN, K_OUT
  
  ! 3. All Real variables MUST be DOUBLE PRECISION to match your F23 code
  DOUBLE PRECISION :: LNP1P2, LNP2P3, LNP1P3
  DOUBLE PRECISION :: LNPU1P(PLVLS - 1), LNPU2P(PLVLS - 2)
  DOUBLE PRECISION :: PDWN, PMID, PUP, SDWN, SMID, SUP, QDWN, QMID, QUP
  DOUBLE PRECISION :: PRES(PLVLS)

  ! 4. The arrays using the passed-in NI and NJ
  DOUBLE PRECISION, INTENT(IN)  :: SSFC(NI, NJ), PSFC(NI, NJ)
  !DOUBLE PRECISION, INTENT(IN)  :: THTA(KOUT)
  DOUBLE PRECISION, INTENT(OUT) :: STHTA(NI, NJ, KOUT)

  DATA PRES/100000.D0, 92500.D0, 85000.D0, 70000.D0,  60000.D0,50000.D0, 40000.D0, &
       30000.D0, 25000.D0, 20000.D0, 15000.D0, 10000.D0, 7000.D0, 5000.D0, &
       3000.D0, 2000.D0, 1000.D0/
  
  DO   K_IN = 1, PLVLS - 2
     LNPU1P(K_IN) = LOG(PRES (K_IN + 1) / PRES(K_IN))
     LNPU2P(K_IN) = LOG(PRES (K_IN + 2) / PRES(K_IN))
  end do
  
  LNPU1P(PLVLS - 1) = LOG(PRES(PLVLS) / PRES(PLVLS - 1))

  
  do 500 k_out = 1, KOUT
     do 400 j = 1,nj
        do 300 i = 1, ni
           if (pthta(i,j,k_out)  .le. 0.) then
              sthta(i,j,k_out) = -9999.
           else if (abs(pthta(i,j,k_out) - psfc(i,j)) .lt. 0.01) then
              sthta(i,j,k_out) = ssfc(i,j)
           else
              k_in = 0
100           continue
              k_in = k_in + 1
              if (k_in .le. plvls) then
                 if (abs(pthta(i,j,k_out) - pres(k_in)) .lt. 0.01) then
                    STHTA (I, J, K_OUT) = SPRES (I, J, K_IN)
                    goto 300
                 ELSE IF (PTHTA (I, J, K_OUT) .GT. PRES (K_IN) ) THEN
                    IF (K_IN .EQ. 1) THEN   
                       PDWN = PSFC (I, J)
                       SDWN = SSFC (I, J)
                       IF (ABS (PSFC (I, J) - PRES (K_IN)) .lt. 0.01) then
                          PMID = PRES (K_IN)
                          PUP = PRES (K_IN + 1)
                          SMID = SPRES (I, J, K_IN)
                          SUP = SPRES (I, J, K_IN + 1)
                          LNP1P2 = LNPU1P(K_IN)
                          LNP1P3 = LOG (PUP / PDWN)
                          LNP2P3 = LOG (PMID / PDWN)
                       else
                          PMID = PRES (K_IN + 1)
                          PUP = PRES (K_IN + 2)
                          SMID = SPRES (I, J, K_IN + 1)
                          SUP = SPRES (I, J, K_IN + 2)
                          LNP1P2 = LNPU1P(K_IN + 1)
                          LNP1P3 = LNPU2P(K_IN)
                          LNP2P3 = LNPU1P(K_IN)
                       end if
                    ELSE IF (K_IN .EQ. PLVLS) THEN
                       PDWN = PRES(K_IN - 2)
                       PMID = PRES(K_IN - 1)
                       PUP = PRES(K_IN)
                       SDWN = SPRES(I, J, K_IN - 2)
                       SMID = SPRES(I, J, K_IN - 1)
                       SUP = SPRES(I, J, K_IN)
                       LNP1P2 = LNPU1P (K_IN - 1)
                       LNP1P3 = LNPU2P (K_IN - 2)
                       LNP2P3 = LNPU1P (K_IN - 2)
                    ELSE IF (PSFC (I, J) .LT. PRES (K_IN - 1) ) THEN
                       PDWN = PSFC (I, J)
                       SDWN = SSFC (I, J)
                       IF (ABS (PSFC (I, J) - PRES (K_IN)) .lt. 0.001) then 
                          PMID = PRES (K_IN)
                          PUP = PRES (K_IN + 1)
                          SMID = SPRES (I, J, K_IN)
                          SUP = SPRES (I, J, K_IN + 1)
                          LNP1P2 = LNPU1P (K_IN)
                          LNP1P3 = LOG (PUP /PDWN)
                          LNP2P3 = LOG (PMID /PDWN)
                       ELSE
                          PMID = PRES (K_IN + 1)
                          PUP = PRES (K_IN + 2)
                          SMID = SPRES (I, J, K_IN + 1)
                          SUP = SPRES (I, J, K_IN + 2)
                          LNP1P2 = LNPU1P (K_IN + 1)
                          LNP1P3 = LNPU2P (K_IN)
                          LNP2P3 = LNPU1P (K_IN)
                       END IF
                    else
                       PDWN = PRES (K_IN - 1)
                       PMID = PRES (K_IN)
                       PUP = PRES (K_IN + 1)
                       SDWN = SPRES (I, J, K_IN - 1)
                       SMID = SPRES (I, J, K_IN)
                       SUP = SPRES (I, J, K_IN + 1)
                       LNP1P2 = LNPU1P (K_IN)
                       LNP1P3 = LNPU2P (K_IN - 1)
                       LNP2P3 = LNPU1P (K_IN - 1)
                    end if
                    goto 200
                 end if
                 goto 100
              end if
200           continue
              QDWN = LOG (PTHTA (I, J, K_OUT) /PMID) * &
                   LOG (PTHTA (I, J, K_OUT) /PUP)/LNP2P3 /LNP1P3
              QMID = -LOG (PTHTA (I, J, K_OUT) /PDWN) * &
                   LOG (PTHTA (I, J, K_OUT) /PUP)/LNP2P3 /LNP1P2
              QUP = LOG (PTHTA (I, J, K_OUT) /PDWN) * &
                   LOG (PTHTA (I, J, K_OUT) /PMID)/LNP1P3 /LNP1P2
              STHTA (I, J, K_OUT) = QDWN * SDWN + QMID *SMID + QUP *SUP
           end if
300        continue
400        continue
500        continue
           return
         end SUBROUTINE S2THTA_OLD
         

 
