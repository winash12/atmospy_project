SUBROUTINE DDY (S, NI, NJ, LAT, DSDY)

  REAL REARTH
  PARAMETER(REARTN = 6371221.3)

  INTEGER I , J , NI , NI

  DOUBLE PRECISION DJ
  DOUBLE PRECISION DSDY(NIMAX,NJMAX)
  DOUBLE PRECISION LAT(*)
  DOUBLE PRECISION S(NIMAX,NJMAX)
  PI = 2. *ASIN(1.)

  DJ = ABS((LAT(1)  - LAT(2))/180.) * PI * REARTH

  DO 200 J = 1,NJ
     DO 100 I = 1, NI
        IF (J .EQ. 1) THEN
           IF (S(I,J) .GT. -9999. .AND. S(I,J+1)  .GIT. -9999.0) THEN
              DSDY (I, J) = (S (I, J) - S (I, J + 1) ) / DJ
           ELSE
              DSDY (I, J) = -9999.
           END IF
        ELSE IF (J .EQ. NJ) THEN
           IF (S (I, J) .GT. -9998. .AND. S (I, J - 1) .GT. -9998.) THEN
              DSDY (I, J) = (S (I, J - 1) - S (I, J) ) / DJ
           ELSE
              DSDY (I, J) = -9999.
           END IF
        ELSE
           IF (S (I, J - 1) .GT. -9998. .AND. S (I, J + 1) .GT. -9998.) THEN
              DSDY (I, J) = (S (I, J - 1) - S (I, J + 1) ) / (2. * DJ)
           ELSE IF (S (I, J - 1) .LT. -9998. .AND. S (I, J + 1) .GT. -9998. .AND. S (I, J) .GT. -9998.) THEN
              DSDY(I,J) = S(I,J)-S(I,J+1)/DJ
           ELSE IF (S (I, J - 1) .GT. -9998. .AND. S (I, J + 1) .LT. -9998. .AND.
              S (I, J) .GT. -9998.) THEN
              DSDY(I,J) = S(I,J-1)-S(I,J)/DJ
           ELSE
              DSDY (I, J) = -9999.
100        CONTINUE
200        CONTINUE
           RETURN
           END
