PROGRAM bit_by_bit_comparison
    USE interp_mod, ONLY: s2thta_f23, f23_dp => dp 
    USE, INTRINSIC :: iso_fortran_env, ONLY: dp => real64
    IMPLICIT NONE

    ! Dimensions aligned with your data info (144, 73, 17 isobaric -> 16 isentropic)
    INTEGER, PARAMETER :: ni = 144, nj = 73, kthta = 16, k_in = 17
    REAL(dp) :: pres(16) = [100000.0_dp, 92500.0_dp, 85000.0_dp, 70000.0_dp, &
                            50000.0_dp, 40000.0_dp, 30000.0_dp, 25000.0_dp, &
                            20000.0_dp, 15000.0_dp, 10000.0_dp, 7000.0_dp,  &
                            5000.0_dp, 3000.0_dp, 2000.0_dp, 1000.0_dp]
    ! Inputs
    REAL(dp) :: ssfc(ni, nj), psfc(ni, nj)
    REAL(dp) :: spres(ni, nj, k_in), pthta(ni, nj, kthta)
    REAL(dp) :: thta(kthta) ! Unused but matches original signature
    
    ! Outputs
    REAL(dp) :: sthta_old(ni, nj, 50) ! Matching MAXLVL=50 in legacy
    REAL(dp) :: sthta_new(ni, nj, kthta)
    
    INTEGER :: i, j, k, mismatch_count
    LOGICAL :: identical
    CALL RANDOM_SEED()
    CALL RANDOM_NUMBER(ssfc)
    CALL RANDOM_NUMBER(psfc)
    CALL RANDOM_NUMBER(spres)
    CALL RANDOM_NUMBER(pthta)
    ssfc = ssfc * 0.1_dp + 1.0_dp
    psfc = psfc * 100000.0_dp  ! Scale to pressure values
    pthta = pthta * 100000.0_dp

     ! 2. Run the legacy F77 code (compiled as S2THTA_OLD)
    PRINT *, "Running Legacy F77 Routine..."
    CALL S2THTA_OLD(ni, nj, kthta, ssfc, psfc, spres, thta, pthta, sthta_old)

    ! 3. Run the modern F2023 code
    PRINT *, "Running Modern F2023 Routine..."
    CALL s2thta_f23(ni, nj, kthta, ssfc, psfc, spres, pres, pthta, sthta_new)
    PRINT *, "--- First 5 Results Comparison ---"
    PRINT '(A15, A20, A20)', "Index (i,j,k)", "Old (Legacy)", "New (Modern)"
    DO k = 1, 1
       DO j = 1, 1
          DO i = 1, 5
             PRINT '(A,3I3,A, F20.10, F20.10)', &
                  "(", i, j, k, ")", sthta_old(i,j,k), sthta_new(i,j,k)
          END DO
       END DO
    END DO
    STOP
    ! 4. Bitwise comparison
    mismatch_count = 0
    PRINT *, "Starting Bitwise Verification..."
    DO k = 1, kthta
        DO j = 1, nj
            DO i = 1, ni
                ! TRANSFER allows us to compare the raw bits of the reals
                IF (TRANSFER(sthta_old(i,j,k), 1_8) /= TRANSFER(sthta_new(i,j,k), 1_8)) THEN
                    mismatch_count = mismatch_count + 1
                    IF (mismatch_count <= 5) THEN
                        PRINT '(A,3I4,A,Z16,A,Z16)', "Mismatch at ", i, j, k, &
                              " | Old: ", sthta_old(i,j,k), " | New: ", sthta_new(i,j,k)
                    END IF
                END IF
            END DO
        END DO
    END DO
    IF (mismatch_count == 0) THEN
       PRINT *, "SUCCESS: Both versions are bit-for-bit identical!"
    ELSE
       PRINT '(A,I10,A)', "FAILURE: Found ", mismatch_count, " bitwise differences."
    END IF

END PROGRAM bit_by_bit_comparison
