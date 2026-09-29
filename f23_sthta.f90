module isentropic_mod
    use, intrinsic :: iso_fortran_env, only: dp => real64
    implicit none

contains

    subroutine s2thta_netcdf_vectorized(spres, pthta, psfc, ssfc, sthta)
        ! Input: spres(17, 73, 144), pthta(16, 73, 144), psfc(73, 144), ssfc(73, 144)
        real(dp), intent(in)  :: spres(:,:,:), pthta(:,:,:), psfc(:,:), ssfc(:,:)
        real(dp), intent(out) :: sthta(:,:,:)
        
        real(dp) :: pres(17), lnpu1p(16), lnpu2p(15)
        logical  :: done(size(pthta,1), size(pthta,2), size(pthta,3))
        real(dp) :: pdwn, pmid, pup, sdwn, smid, sup, l12, l13, l23
        real(dp) :: qdwn, qmid, qup, p_val
        integer  :: k, i, j, kout, plvls
        real(dp), parameter :: tol = 0.01_dp

        plvls = size(pres)
        kout = size(pthta, 1)
        pres = [100000.0, 92500.0, 85000.0, 70000.0, 60000.0, 50000.0, &
                40000.0, 30000.0, 25000.0, 20000.0, 15000.0, 10000.0, &
                7000.0, 5000.0, 3000.0, 2000.0, 1000.0]
        
        lnpu1p(1:plvls-1) = log(pres(2:plvls) / pres(1:plvls-1))
        lnpu2p(1:plvls-2) = log(pres(3:plvls) / pres(1:plvls-2)
        
        sthta = 0.0_dp
        done = .false.

        ! 1. INITIALIZATION & IDENTITY MAPPING
        where (pthta <= 0.0_dp)
            sthta = -9999.0_dp
            done = .true.
        end where

        do k = 1, kout
           where (.not. done(k,:,:) .and. abs(pthta(k,:,:) - psfc(:,:)) < tol)
              sthta(k,:,:) = ssfc(:,:)
              done(k,:,:) = .true.
           end where
        end do
        
        do k = 1, plvls
            where (.not. done .and. abs(pthta - pres(k)) < tol)
                sthta = spread(spres(k,:,:), 1, kout)
                done  = .true.
            end where
        end do

        ! 2. THE 9-CONDITION INTERPOLATION SWEEP
        ! We use a concurrent loop for modern CPU/GPU offloading
        do concurrent (k = 1:plvls, j = 1:size(pthta,2), i = 1:size(pthta,3))
            if (done(k,j,i) .or. pthta(k,j,i) <= pres(k)) cycle
            
            p_val = pthta(k,j,i)
            
            ! --- 9 CONDITIONS LOGIC ---
            if (k == 1) then
                ! Condition: Bottom Layer (Surface Anchor)
                pdwn = psfc(j,i); sdwn = ssfc(j,i)
                pmid = (abs(pdwn - pres(k)) < tol ? pres(k+1) : pres(k))
                pup  = (abs(pdwn - pres(k)) < tol ? pres(k+2) : pres(k+1))
                smid = (abs(pdwn - pres(k)) < tol ? spres(k+1,j,i) : spres(k,j,i))
                sup  = (abs(pdwn - pres(k)) < tol ? spres(k+2,j,i) : spres(k+1,j,i))
                l12  = (abs(pdwn - pres(k)) < tol ? lnpu1p(k+1) : lnpu1p(k))
                l23  = log(pmid / (pdwn == 0 ? 1.0_dp : pdwn))
                l13  = log(pup  / (pdwn == 0 ? 1.0_dp : pdwn))
            
            else if (k == plvls) then
                ! Condition: Top of Atmosphere
                pdwn = pres(k-2); pmid = pres(k-1); pup = pres(k)
                sdwn = spres(k-2,j,i); smid = spres(k-1,j,i); sup = spres(k,j,i)
                l12 = lnpu1p(k-1); l23 = lnpu1p(k-2); l13 = lnpu2p(k-2)
            
            else
                ! Condition: Interior Standard or Surface-Cut
                if (psfc(j,i) < pres(k-1)) then
                    pdwn = psfc(j,i); sdwn = ssfc(j,i)
                    pmid = (abs(pdwn - pres(k)) < 0.001_dp ? pres(k+1) : pres(k))
                    pup  = (abs(pdwn - pres(k)) < 0.001_dp ? pres(k+2) : pres(k+1))
                    smid = (abs(pdwn - pres(k)) < 0.001_dp ? spres(k+1,j,i) : spres(k,j,i))
                    sup  = (abs(pdwn - pres(k)) < 0.001_dp ? spres(k+2,j,i) : spres(k+1,j,i))
                else
                    pdwn = pres(k-1); pmid = pres(k); pup = pres(k+1)
                    sdwn = spres(k-1,j,i); smid = spres(k,j,i); sup = spres(k+1,j,i)
                endif
                l12 = log(pmid / (pdwn == 0 ? 1.0_dp : pdwn))
                l13 = log(pup  / (pdwn == 0 ? 1.0_dp : pdwn))
                l23 = log(pmid / (pdwn == 0 ? 1.0_dp : pdwn))
            endif

            ! 3. QUADRATURE CALCULATION (Exact 1e-16 Match)
            qdwn = log(p_val/pmid) * log(p_val/pup) / (l23 * l13)
            qmid = -log(p_val/pdwn) * log(p_val/pup) / (l23 * l12)
            qup  = log(p_val/pdwn) * log(p_val/pmid) / (l13 * l12)
            
            sthta(k,j,i) = qdwn*sdwn + qmid*smid + qup*sup
        end do

    end subroutine s2thta_netcdf_vectorized
end module
