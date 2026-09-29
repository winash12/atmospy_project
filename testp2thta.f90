PROGRAM test_comparison
    USE grid_mod
    IMPLICIT NONE
    ! ... [Declare all variables ni, nj, lat, lon, tsfc, psfc, tpres, thta, etc.] ...

    ! 1. Call the Modern Routine
    CALL p2thta(ni, nj, lat, lon, tsfc, psfc, tpres, kthta, thta, pthta_new)

    ! 2. Call the Old Routine (compile this as a separate object file)
    CALL p2thta_old(ni, nj, lat, lon, tsfc, psfc, tpres, kthta_old, thta_old, pthta_old)

    ! 3. Bit-by-bit Comparison
    print *, "--- BIT-BY-BIT COMPARISON RESULTS ---"
    diff_count = 0
    DO k = 1, kthta
        DO j = 1, nj
            DO i = 1, ni
                IF (TRANSFER(pthta_new(i,j,k), 0_4) /= TRANSFER(pthta_old(i,j,k), 0_4)) THEN
                    print *, "Mismatch at (", i, j, k, "): New=", pthta_new(i,j,k), " Old=", pthta_old(i,j,k)
                    diff_count = diff_count + 1
                END IF
            END DO
        END DO
     END DO

     IF (diff_count == 0) THEN
        print *, "SUCCESS: Bit-for-bit identical!"
    ELSE
        print *, "FAILURE: Found ", diff_count, " differences."
    END IF
  END PROGRAM
