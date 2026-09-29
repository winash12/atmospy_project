module interp_mod
  use, intrinsic :: iso_fortran_env, only: dp => real64
  implicit none
  
  
contains
  
  subroutine s2thta_f23(ni, nj, kthta, ssfc, psfc, spres, pres, pthta, sthta)
    integer, intent(in) :: ni, nj, kthta
    real(dp), intent(in) :: ssfc(ni,nj), psfc(ni,nj), spres(ni,nj,*)
    real(dp), intent(in) :: pres(*), pthta(ni,nj,kthta)
    real(dp), intent(out) :: sthta(ni,nj,kthta)
    
    ! Exact replication of F77 intermediate variables
    real(dp) :: lnpu1p(15), lnpu2p(14) 
    real(dp) :: lnp1p2, lnp1p3, lnp2p3
    real(dp) :: pdwn, pmid, pup, sdwn, smid, sup, qdwn, qmid, qup
    integer  :: i, j, k, kin, m, plvls
    
    plvls = 16
    ! Replicate F77 pre-calculation loop exactly
    do m = 1, plvls - 2
       lnpu1p(m) = log(pres(m + 1) / pres(m))
       lnpu2p(m) = log(pres(m + 2) / pres(m))
    end do
    
    lnpu1p(plvls - 1) = log(pres(plvls) / pres(plvls - 1))
    
    sthta = -9999.0_dp
    
    do k = 1, kthta
       do j = 1, nj
          do i = 1, ni
             if (pthta(i,j,k) <= 0.0_dp) cycle
             
             ! Exact Surface Match logic from F77
             if (abs(pthta(i,j,k) - psfc(i,j)) < 0.01_dp) then
                sthta(i,j,k) = ssfc(i,j)
                cycle
             end if
             kin = 1
             do m = 1, plvls
                if (pthta(i,j,k) > pres(m)) then
                   kin = m
                   exit
                end if
             end do
             if (kin == 1) then
                pdwn = psfc(i,j)
                sdwn = ssfc(i,j)
                if (abs(psfc(i,j) - pres(1)) < 0.01_dp) then
                   pmid = pres(1)
                   pup  = pres(2)
                   smid = spres(i,j,1)
                   sup  = spres(i,j,2)
                   lnp1p2 = lnpu1p(1)
                   lnp1p3 = log(pup / pdwn)
                   lnp2p3 = log(pmid / pdwn)
                else
                   pmid = pres(2)
                   pup  = pres(3)
                   smid = spres(i,j,2)
                   sup  = spres(i,j,3)
                   lnp1p2 = lnpu1p(2)
                   lnp1p3 = lnpu2p(1)
                   lnp2p3 = lnpu1p(1)
                end if
             else if (kin == plvls) then
                pdwn = pres(kin-2)
                pmid = pres(kin-1)
                pup  = pres(kin)
                sdwn = spres(i,j,kin-2)
                smid = spres(i,j,kin-1)
                sup  = spres(i,j,kin)
                lnp1p2 = lnpu1p(kin-1)
                lnp1p3 = lnpu2p(kin-2)
                lnp2p3 = lnpu1p(kin-2)
                else
                   pdwn = pres(kin-1)
                   pmid = pres(kin)
                   pup  = pres(kin+1)
                   sdwn = spres(i,j,kin-1)
                   smid = spres(i,j,kin)
                   sup  = spres(i,j,kin+1)
                   lnp1p2 = lnpu1p(kin)
                   lnp1p3 = lnpu2p(kin-1)
                   lnp2p3 = lnpu1p(kin-1)
                end if
                qdwn = log(pthta(i,j,k) / pmid) * log(pthta(i,j,k) / pup) / (lnp2p3 * lnp1p3)
                qmid = -log(pthta(i,j,k) / pdwn) * log(pthta(i,j,k) / pup) / (lnp2p3 * lnp1p2)
                qup  = log(pthta(i,j,k) / pdwn) * log(pthta(i,j,k) / pmid) / (lnp1p3 * lnp1p2)

                sthta(i,j,k) = qdwn * sdwn + qmid * smid + qup * sup
             end do
          end do
        end do
      end subroutine s2thta_f23
    end module interp_mod
