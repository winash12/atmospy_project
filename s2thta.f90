SUBROUTINE S2THTA (NI, NJ, KTHTA, SSFC, PSFC, SPRES, THTA, PTHTA,STHTA)
  
  INTEGER MAXLVL
  PARAMETER (MAXLVL = 50)
  INTEGER PLVLS
  PARAMETER (PLVLS = 16)
  INTEGER I
  INTEGER J
  INTEGER KIN
  INTEGER KOUT
  INTEGER KTHTA
  INTEGER NI 
  INTEGER NJ
  REAL LNPIP2

  REAL LNP1P2 , LNP2P3, LNPU1P (PLVLS - 1), LNPU2P (PLVLS - 2)
  REAL PDWN,PMID , PRES(PLVLS), PSFC(NIMAX,NJMAX)
  REAL PTHTA (NIMAX, NJMAX, KTHTA)
  REAL PUP
  REAL QDWN
  REAL QMID,QUP
  REAL SDWN,SMID
  REAL SSFC (NIMAX, NJMAX)
  REAL SPRES (NIMAX, NJMAX, KTHTA)
  REAL STHTA (NIMAX, NJMAX, MAXLVL)

  REAL SUP,THTA (*)
  DATA PRES/100000., 92500., 85000., 70000., 50000., 40000., &
       30000., 25000., 20000., 15000., 10000., 7000.,5000., 3000., 2000., &
       1000./
  DO   KIN = 1, PLVLS - 2
     LNPU1P(KIN) = ALOG(PRES (KIN + 1) / PRES(KIN))
     LNPU2P(KIN) = ALOG(PRES (KIN + 2) / PRES(KIN))
  end do
  LNPU1P(PLVLS - 1) = ALOG(PRES(PLVLS) / PRES(PLVLS - 1))
  do 500 kout = 1, kthta
     do 400 j = 1,nj
        do 300 i = 1, ni
           if (pthta(i,j,kout)  .le. 0.) then
              sthta(i,j,kout) = -9999.
           else if (abs(pthta(i,j,kout) - psfc(i,j)) .lt. 0.01) then
              sthta(i,j,k) = ssfc(i,j)
           else
              kin = 0
100           continue
              kin = kin + 1
              if (kin .le. plvls) then
                 if (abs(pthta(i,j,kout) - pres(kin)) .lt. 0.01) then
                    STHTA (I, J, KOUT) = SPRES (I, J, KIN)
                    goto 300
                 ELSE IF (PTHTA (I, J, KOUT) .GT. PRES (KIN) ) THEN
                    IF (KIN .EQ. 1) THEN   
                       PDWN = PSFC (I, J)
                       SDWN = SSFC (I, J)
                       IF (ABS (PSFC (I, J) - PRES (KIN)) .lt. 0.01) then
                          PMID = PRES (KIN)
                          PUP = PRES (KIN + 1)
                          SMID = SPRES (I, J, KIN)
                          SUP = SPRES (I, J, KIN + 1)
                          LNP1P2 = LNPU1P(KIN)
                          LNP1P3 = ALOG (PUP / PDWN)
                          LNP2P3 = ALOG (PMID / PDWN)
                       else
                          PMID = PRES (KIN + 1)
                          PUP = PRES (KIN + 2)
                          SMID = SPRES (I, J, KIN + 1)
                          SUP = SPRES (I, J, KIN + 2)
                          LNPIP2 = LNPU1P(KIN + 1)
                          LNP1P3 = LNPU2P(KIN)
                          LNP2P3 = LNPU1P(KIN)
                       end if
                    ELSE IF (KIN .EQ. PLVLS) THEN
                       PDWN = PRES(KIN - 2)
                       PMID = PRES(KIN - 1)
                       PUP = PRES(KIN)
                       SDWN = SPRES(I, J, KIN - 2)
                       SMID = SPRES(I, J, KIN - 1)
                       SUP = SPRES(I, J, KIN)
                       LNPIP2 = LNPU1P (KIN - 1)
                       LNPIP3 = LNPU2P (KIN - 2)
                       LNP2P3 = LNPU1P (KIN - 2)
                    ELSE IF (PSFC (I, J) .LT. PRES (KIN - 1) ) THEN
                       PDWN = PSFC (I, J)
                       SDWN = SSFC (I, J)
                       IF (ABS (PSFC (I, J) - PRES (KIN)) .lt. 0.001) then 
                          PMID = PRES (KIN)
                          PUP = PRES (KIN + 1)
                          SMID = SPRES (I, J, KIN)
                          SUP = SPRES (I, J, KIN + 1)
                          LNPlP2 = LNPU1P (KIN)
                          LNPlP3 = ALOG (PUP /PDWN)
                          LNP2P3 = ALOG (PMID /PDWN)
                       ELSE
                          PMID = PRES (KIN + 1)
                          PUP = PRES (KIN + 2)
                          SMID = SPRES (I, J, KIN + 1)
                          SUP = SPRES (I, J, KIN + 2)
                          LNPlP2 = LNPU1P (KIN + 1)
                          LNPlP3 = LNPU2P (KIN)
                          LNP2P3 = LNPU1P (KIN)
                       END IF
                    else
                       PDWN = PRES (KIN - 1)
                       PMID = PRES (KIN)
                       PUP = PRES (KIN + 1)
                       SDWN = SPRES (I, J, KIN - 1)
                       SMID = SPRES (I, J, KIN)
                       SUP = SPRES (I, J, KIN + 1)
                       LNPlP2 = LNPU1P (KIN)
                       LNP1P3 = LNPU2P (KIN - 1)
                       LNP2P3 = LNPU1P (KIN - 1)
                    end if
                    goto 200
                 end if
                 goto 100
              end if
200           continue
              QDWN = ALOG (PTHTA (I, J, KOUT) /PMID) * &
                   ALOG (PTHTA (I, J, KOUT) /PUP)/LNP2P3 /LNPlP3
              QMID = -ALOG (PTHTA (I, J, KOUT) /PDWN) * &
                   ALOG (PTHTA (I, J, KOUT) /PUP)/LNP2P3 /LNPlP2
              QUP = ALOG (PTHTA (I, J, KOUT) /PDWN) * &
                   ALOG (PTHTA (I, J, KOUT) /PMID)/LNPlP3 /LNP1P2
              STHTA (I, J, KOUT) = QDWN * SDWN + QMID *SMID + QUP *SUP
           end if
300        continue
400        continue
500        continue
           return
         end SUBROUTINE S2THTA


