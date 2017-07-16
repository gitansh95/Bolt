#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import arrayfire as af

def f_interp_2d(self, dt):
  # Since the interpolation function are being performed in position space,
  # the arrays used in the computation need to be in positionsExpanded form.

  # Obtaining the left-bottom corner coordinates
  # (lowest values of the canonical coordinates in the local zone)
  # Additionally, we also obtain the size of the local zone
  ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()
  
  q1_center_new = self.q1_center - self._A_q1*dt
  q2_center_new = self.q2_center - self._A_q2*dt

  # Obtaining the center coordinates:
  (i_q1_center, i_q2_center) = (i_q1_lowest + 0.5, i_q2_lowest + 0.5)

  # Obtaining the left, and bottom boundaries for the local zones:
  q1_boundary_lower = self.q1_start + i_q1_center * self.dq1
  q2_boundary_lower = self.q2_start + i_q2_center * self.dq2

  # Adding N_ghost to account for the offset due to ghost zones:
  q1_interpolant = (q1_center_new - q1_boundary_lower)/self.dq1 + self.N_ghost
  q2_interpolant = (q2_center_new - q2_boundary_lower)/self.dq2 + self.N_ghost

  self.f = af.approx2(self.f,\
                      q1_interpolant,\
                      q2_interpolant,\
                      af.INTERP.BICUBIC_SPLINE
                     )

  af.eval(self.f)
  return


def f_interp_vel_3d(self, dt):
  # Since the interpolation function are being performed in velocity space,
  # the arrays used in the computation need to be in velocitiesExpanded form.
  # Hence we will need to convert the same:

  p1 = self._convert(self.p1)
  p2 = self._convert(self.p2)
  p3 = self._convert(self.p3)

  q1 = self._convert(self.q1_center)
  q2 = self._convert(self.q2_center)

  E1 = self._convert(self.E1)
  E2 = self._convert(self.E2)
  E3 = self._convert(self.E3)
  B1 = self._convert(self.B1)
  B2 = self._convert(self.B2)
  B3 = self._convert(self.B3)

  params = self.physical_system.params

  p1_new = p1 - 0.5 * dt * self._A_p(q1, q2, p1, p2, p3, E1, E2, E3, B1, B2, B3, params)[0]
  p2_new = p2 - dt * self._A_p(q1, q2, p1, p2, p3, E1, E2, E3, B1, B2, B3, params)[1]
  p3_new = p3 - dt * self._A_p(q1, q2, p1, p2, p3, E1, E2, E3, B1, B2, B3, params)[2]

  # Transforming interpolant to go from [0, N_p - 1]:
  p1_lower_boundary = self.p1_start + 0.5 * self.dp1
  p2_lower_boundary = self.p2_start + 0.5 * self.dp2
  p3_lower_boundary = self.p3_start + 0.5 * self.dp3

  p1_interpolant = (p1_new - p1_lower_boundary)/self.dp1
  p2_interpolant = (p2_new - p2_lower_boundary)/self.dp2
  p3_interpolant = (p3_new - p3_lower_boundary)/self.dp3

  # We perform the 3d interpolation by performing individual 1d + 2d interpolations:
  # Reordering to bring the variation in values along axis 0 and axis 1

  self.f = af.approx1(af.reorder(self.f),\
                      af.reorder(p1_interpolant),\
                      af.INTERP.CUBIC_SPLINE
                     )

  self.f = af.approx2(af.reorder(self.f, 2, 3, 1, 0),\
                      af.reorder(p2_interpolant, 2, 3, 0, 1),\
                      af.reorder(p3_interpolant, 2, 3, 0, 1),\
                      af.INTERP.BICUBIC_SPLINE
                    )

  self.f = af.approx1(af.reorder(self.f, 3, 2, 1, 0),\
                      af.reorder(p1_interpolant),\
                      af.INTERP.CUBIC_SPLINE
                     )

  self.f = af.reorder(self.f)

  af.eval(self.f)
  return