subroutine s2thta_vector(ni, nj, kin, kout, pres, ssfc, psfc, spres, pthta, sthta)
  implicit none
  integer, intent(in) :: ni, nj, kin, kout
  double precision, intent(in)  :: spres(ni, nj, kin), pthta(ni, nj, kout)
  double precision, intent(in)  :: ssfc(ni, nj), psfc(ni, nj)
  double precision, intent(in)  :: pres(:)
  double precision, intent(out) :: sthta(ni, nj, kout)

  integer :: k, plvls
  logical :: done(ni, nj, kout), active(ni, nj, kout)
  logical :: m_psfc(ni, nj, kout), c_mask(ni, nj, kout)
  
  ! Temporary 3D variables for "broadcasting"
  double precision :: psfc_3d(ni, nj, kout), ssfc_3d(ni, nj, kout)
  double precision :: pdwn(ni, nj, kout), pmid(ni, nj, kout), pup(ni, nj, kout)
  double precision :: sdwn(ni, nj, kout), smid(ni, nj, kout), sup(ni, nj, kout)
  double precision :: l12(ni, nj, kout), l13(ni, nj, kout), l23(ni, nj, kout)
  double precision :: lnpu1p(size(pres)-1), lnpu2p(size(pres)-2)

  plvls = size(pres)

  ! 1. Precompute logs (Scalar operations)
  do k = 1, plvls - 2
     lnpu1p(k) = log(pres(k+1) / pres(k))
     lnpu2p(k) = log(pres(k+2) / pres(k))
  end do
  lnpu1p(plvls-1) = log(pres(plvls) / pres(plvls-1))

  ! 2. Broadcast 2D Surface arrays to 3D (Like NumPy's auto-broadcast)
  psfc_3d = spread(psfc, 3, kout)
  ssfc_3d = spread(ssfc, 3, kout)

  sthta = -9999.0d0
  done  = .false.

  ! 3. Initial Masks (Handled in 3D chunks)
  where (pthta <= 0.0d0) done = .true.
  where (.not. done .and. abs(pthta - psfc_3d) < 0.01d0)
     sthta = ssfc_3d
     done  = .true.
  end where

  ! 4. Exact Match (3D)
  do k = 1, plvls
     where (.not. done .and. abs(pthta - pres(k)) < 0.01d0)
        sthta = spread(spres(:,:,k), 3, kout) ! Slice broadcasting
        done  = .true.
     end where
  end do

  ! 5. MAIN SWEEP: Single loop over input levels (K)
  do k = 1, plvls
     active = (.not. done .and. pthta > pres(k))
     if (.not. any(active)) cycle

     if (k == 1) then
        c_mask = (active .and. abs(psfc_3d - pres(k)) < 0.01d0)
        where (c_mask)
           pdwn = psfc_3d; sdwn = ssfc_3d
           pmid = pres(k); smid = spread(spres(:,:,k), 3, kout)
           pup  = pres(k+1); sup = spread(spres(:,:,k+1), 3, kout)
           l12  = lnpu1p(k); l23 = log(pmid/pdwn); l13 = log(pup/pdwn)
        elsewhere (active)
           pdwn = psfc_3d; sdwn = ssfc_3d
           pmid = pres(k+1); smid = spread(spres(:,:,k+1), 3, kout)
           pup  = pres(k+2); sup  = spread(spres(:,:,k+2), 3, kout)
           l12  = lnpu1p(k+1); l23 = lnpu1p(k); l13 = lnpu2p(k)
        end where
     else if (k == plvls) then
        where (active)
           pdwn = pres(k-2); sdwn = spread(spres(:,:,k-2), 3, kout)
           pmid = pres(k-1); smid = spread(spres(:,:,k-1), 3, kout)
           pup  = pres(k);   sup  = spread(spres(:,:,k), 3, kout)
           l12 = lnpu1p(k-1); l23 = lnpu1p(k-2); l13 = lnpu2p(k-2)
        end where
     else
        m_psfc = (active .and. psfc_3d < pres(k-1))
        c_mask = (m_psfc .and. abs(psfc_3d - pres(k)) < 0.001d0)
        
        where (c_mask)
           pdwn = psfc_3d; sdwn = ssfc_3d
           pmid = pres(k); smid = spread(spres(:,:,k), 3, kout)
           pup  = pres(k+1); sup = spread(spres(:,:,k+1), 3, kout)
           l12 = lnpu1p(k); l23 = log(pmid/pdwn); l13 = log(pup/pdwn)
        elsewhere (m_psfc)
           pdwn = psfc_3d; sdwn = ssfc_3d
           pmid = pres(k+1); smid = spread(spres(:,:,k+1), 3, kout)
           pup  = pres(k+2); sup  = spread(spres(:,:,k+2), 3, kout)
           l12 = lnpu1p(k+1); l23 = lnpu1p(k); l13 = lnpu2p(k)
        elsewhere (active)
           pdwn = pres(k-1); sdwn = spread(spres(:,:,k-1), 3, kout)
           pmid = pres(k);   smid = spread(spres(:,:,k), 3, kout)
           pup  = pres(k+1); sup  = spread(spres(:,:,k+1), 3, kout)
           l12 = lnpu1p(k); l23 = lnpu1p(k-1); l13 = lnpu2p(k-1)
        end where
     end if

     ! 6. Math Block (Vectorized)
     where (active)
        sthta = (log(pthta/pmid) * log(pthta/pup) / (l23 * l13)) * sdwn + &
                (-log(pthta/pdwn) * log(pthta/pup) / (l23 * l12)) * smid + &
                (log(pthta/pdwn) * log(pthta/pmid) / (l13 * l12)) * sup
        done  = .true.
     end where
  end do
end subroutine s2thta_vector
