PROGRAM bit_for_bit_verify
  USE f2023_module  ! Your modernized code
  IMPLICIT NONE
  
    ! Grid dimensions
  INTEGER, PARAMETER :: NI=10, NJ=10, KTHTA=5, PLVLS=16
  REAL(8) :: PSFC(NI,NJ), SSFC(NI,NJ), PRES(PLVLS)
  REAL(8) :: SPRES(NI,NJ,PLVLS), PTHATA(NI,NJ,KTHTA)
  
  ! Result arrays
  REAL(8) :: STHTA_F77(NI,NJ,KTHTA), STHTA_F23(NI,NJ,KTHTA)
  INTEGER(8) :: BITS_F77, BITS_F23
  INTEGER :: I, J, K, DIFF_COUNT
  
  ! 1. Setup identical test data (Use specific constants for bit-consistency)
  PRES = [100000.0, 92500.0, 85000.0, 70000.0, 50000.0, 40000.0, &
       30000.0, 25000.0, 20000.0, 15000.0, 10000.0, 7000.0, &
       5000.0, 3000.0, 2000.0, 1000.0]
  CALL RANDOM_NUMBER(SPRES)
  CALL RANDOM_NUMBER(PTHATA)
  PSFC = 101325.0
  SSFC = 288.15
  CALL S2THTA_F77(NI, NJ, KTHTA, SSFC, PSFC, SPRES, PRES, PTHATA, STHTA_F77)
  
  ! 3. Execute Modern F23 code 
  CALL S2THTA_F23(NI, NJ, KTHTA, SSFC, PSFC, SPRES, PRES, PTHATA, STHTA_F23)
  
  DIFF_COUNT = 0
  DO K = 1, KTHTA
     DO J = 1, NJ
        DO I = 1, NI
           ! TRANSFER treats the real bits as integer bits
           BITS_F77 = TRANSFER(STHTA_F77(I,J,K), BITS_F77)
           BITS_F23 = TRANSFER(STHTA_F23(I,J,K), BITS_F23)
           
           IF (BITS_F77 /= BITS_F23) THEN
              PRINT '(A,3I3,A,Z16,A,Z16)', "Bit mismatch at ", I,J,K, &
                   " F77:", BITS_F77, " F23:", BITS_F23
              DIFF_COUNT = DIFF_COUNT + 1
           END IF
        END DO
     END DO
  END DO

  IF (DIFF_COUNT == 0) THEN
     PRINT *, "SUCCESS: Bit-for-bit identical results!"
  ELSE
     PRINT *, "FAILURE: ", DIFF_COUNT, " elements differed at the bit level."
  END IF
END PROGRAM bit_for_bit_verify
  
