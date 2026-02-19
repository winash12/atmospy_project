subroutine s2thta_vector(ni, nj, kin, kout, pres,ssfc, psfc, spres, pthta, sthta)
  implicit none
  integer, intent(in) :: ni, nj, kin, kout
  double precision, intent(in)  :: spres(ni, nj, kin), pthta(ni, nj, kout)
  double precision, intent(in)  :: ssfc(ni, nj), psfc(ni, nj)
  double precision, intent(out) :: sthta(ni, nj, kout)



  double precision,intent(in) :: pres(:)
  !data pres / 100000.d0, 92500.d0, 85000.d0, 70000.d0, 60000.d0, 50000.d0, &
  !            40000.d0, 30000.d0, 25000.d0, 20000.d0, 15000.d0, 10000.d0, &
  !            7000.d0, 5000.d0, 3000.d0, 2000.d0, 1000.d0 /
  integer :: plvls
  integer :: k, ko
  logical :: done(ni, nj, kout), active(ni, nj), m_psfc(ni, nj), c_mask(ni, nj)
  double precision :: lnpu1p(size(pres)-1), lnpu2p(size(pres)-2)
  double precision :: pdwn(ni, nj), pmid(ni, nj), pup(ni, nj)
  double precision :: sdwn(ni, nj), smid(ni, nj), sup(ni, nj)
  double precision :: l12(ni, nj), l13(ni, nj), l23(ni, nj)


  plvls = size(pres)
  ! F77 precomputation alignment
  do k = 1, plvls - 2
     lnpu1p(k) = log(pres(k+1) / pres(k))
     lnpu2p(k) = log(pres(k+2) / pres(k))
  end do
  lnpu1p(plvls-1) = log(pres(plvls) / pres(plvls-1))

  sthta = -9999.0d0
  done  = .false.

  ! Condition 1 & 2
  do ko = 1, kout
     where (pthta(:,:,ko) <= 0.0d0)
        done(:,:,ko) = .true.
     end where
     where (.not. done(:,:,ko) .and. abs(pthta(:,:,ko) - psfc) < 0.01d0)
        sthta(:,:,ko) = ssfc
        done(:,:,ko)  = .true.
     end where
  end do

  ! Condition 3: Exact Match
  do k = 1, plvls
     do ko = 1, kout
        where (.not. done(:,:,ko) .and. abs(pthta(:,:,ko) - pres(k)) < 0.01d0)
           sthta(:,:,ko) = spres(:,:,k)
           done(:,:,ko)  = .true.
        end where
     end do
  end do

  ! Main Sweep: Match F77 k_in loop
  do k = 1, plvls
     do ko = 1, kout
        ! This matches F77: ELSE IF (PTHTA > PRES(K_IN))
        active = (.not. done(:,:,ko) .and. pthta(:,:,ko) > pres(k))
        if (.not. any(active)) cycle

        if (k == 1) then
           c_mask = (abs(psfc - pres(k)) < 0.01d0)
           where (active .and. c_mask)
              pdwn = psfc; sdwn = ssfc
              pmid = pres(k); smid = spres(:,:,k)
              pup  = pres(k+1); sup  = spres(:,:,k+1)
              l12  = lnpu1p(k); l23 = log(pmid/pdwn); l13 = log(pup/pdwn)
           end where
           where (active .and. .not. c_mask)
              pdwn = psfc; sdwn = ssfc
              pmid = pres(k+1); smid = spres(:,:,k+1)
              pup  = pres(k+2); sup  = spres(:,:,k+2)
              l12  = lnpu1p(k+1); l23 = lnpu1p(k); l13 = lnpu2p(k)
           end where
        else if (k == plvls) then
           where (active)
              pdwn = pres(k-2); sdwn = spres(:,:,k-2)
              pmid = pres(k-1); smid = spres(:,:,k-1)
              pup  = pres(k);   sup  = spres(:,:,k)
              l12 = lnpu1p(k-1); l23 = lnpu1p(k-2); l13 = lnpu2p(k-2)
           end where
        else
           ! Condition 6: PSFC < PRES(K_IN-1)
           m_psfc = (active .and. psfc < pres(k-1))
           c_mask = (abs(psfc - pres(k)) < 0.001d0)
           
           where (m_psfc .and. c_mask)
              pdwn = psfc; sdwn = ssfc
              pmid = pres(k); smid = spres(:,:,k)
              pup  = pres(k+1); sup  = spres(:,:,k+1)
              l12 = lnpu1p(k); l23 = log(pmid/pdwn); l13 = log(pup/pdwn)
           end where
           where (m_psfc .and. .not. c_mask)
              pdwn = psfc; sdwn = ssfc
              pmid = pres(k+1); smid = spres(:,:,k+1)
              pup  = pres(k+2); sup  = spres(:,:,k+2)
              l12 = lnpu1p(k+1); l23 = lnpu1p(k); l13 = lnpu2p(k)
           end where
           ! Condition 7: Standard Interior
           where (active .and. .not. m_psfc)
              pdwn = pres(k-1); sdwn = spres(:,:,k-1)
              pmid = pres(k);   smid = spres(:,:,k)
              pup  = pres(k+1); sup  = spres(:,:,k+1)
              l12 = lnpu1p(k); l23 = lnpu1p(k-1); l13 = lnpu2p(k-1)
           end where
        end if

        where (active)
           sthta(:,:,ko) = (log(pthta(:,:,ko)/pmid) * log(pthta(:,:,ko)/pup) / (l23 * l13)) * sdwn + &
                           (-log(pthta(:,:,ko)/pdwn) * log(pthta(:,:,ko)/pup) / (l23 * l12)) * smid + &
                           (log(pthta(:,:,ko)/pdwn) * log(pthta(:,:,ko)/pmid) / (l13 * l12)) * sup
           done(:,:,ko)  = .true.
        end where
     end do
  end do
end subroutine s2thta_vector
