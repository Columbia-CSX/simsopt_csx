import unittest
from simsopt.field.coil import Coil, apply_symmetries_to_curves, apply_symmetries_to_currents
from simsopt.geo import CurveFilament, FrameRotation, \
    create_multifilament_grid, ZeroRotation, FramedCurveCentroid, FramedCurveFrenet
from simsopt.configs.zoo import get_ncsx_data

import numpy as np

class FrameRotationTesting(unittest.TestCase):

    def test_rotatedframe_dash(self):
        for centroid in [True, False]:
            for order in [1]:#[None, 1]:
                with self.subTest(order=order):
                    self.subtest_rotatedframe_dash(order, centroid)

    def subtest_rotatedframe_dash(self, order, centroid):
        assert order in [1, None]
        curves, currents, ma = get_ncsx_data(Nt_coils=6, ppp=120)
        c = curves[0]

        if order == 1:
            rotation = FrameRotation(c.quadpoints, order)
            # rotation.x = np.array([0, 0.1, 0.3])
            rotation.x = np.array([0, 0, 0])
            rotationShared = FrameRotation(curves[0].quadpoints, order, dofs=rotation.dofs)
            assert np.allclose(rotation.x, rotationShared.x)
            assert np.allclose(rotation.alpha(c.quadpoints), rotationShared.alpha(c.quadpoints))
        else:
            rotation = ZeroRotation(c.quadpoints)

        if centroid:
            framedcurve = FramedCurveCentroid(c, rotation)
        else:
            framedcurve = FramedCurveFrenet(c, rotation)
        t, n, b = framedcurve.rotated_frame()
        td, nd, bd = framedcurve.rotated_frame_dash()
       
        idx = 16

        dphi = rotation.quadpoints[1]
        weights = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]
        est_t = 0
        est_n = 0
        est_b = 0
        for j in range(-4, 5):
            est_t += weights[j+4] * t[idx+j, :]
            est_n += weights[j+4] * n[idx+j, :]
            est_b += weights[j+4] * b[idx+j, :]
        est_t *= 1./dphi
        est_n *= 1./dphi
        est_b *= 1./dphi
        assert np.all(np.abs(est_t - td[idx]) < 1e-8)
        assert np.all(np.abs(est_n - nd[idx]) < 1e-8)
        assert np.all(np.abs(est_b - bd[idx]) < 1e-8)

    def test_rotatedframe_dashdash(self):
        for centroid in [True]:
            for order in [1]:
                with self.subTest(order=order):
                    self.subtest_rotatedframe_dashdash(order, centroid)

    def subtest_rotatedframe_dashdash(self, order, centroid):
        # assert order in [1, None]
        curves, currents, ma = get_ncsx_data(Nt_coils=6, ppp=120)
        c = curves[0]

        rotation = FrameRotation(c.quadpoints, order)
        rotation.x = np.array([0, 0.1, 0.3])
        rotationShared = FrameRotation(curves[0].quadpoints, order, dofs=rotation.dofs)

        assert np.allclose(rotation.x, rotationShared.x)
        assert np.allclose(rotation.alpha(c.quadpoints), rotationShared.alpha(c.quadpoints))

        framedcurve = FramedCurveCentroid(c, rotation)

        t, n, b = framedcurve.rotated_frame()
        td, nd, bd = framedcurve.rotated_frame_dash()
        tdd, ndd, bdd = framedcurve.rotated_frame_dashdash()
       
        idx = 16

        dphi = rotation.quadpoints[1]
        weights = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]
        est_t = 0
        est_n = 0
        est_b = 0
        for j in range(-4, 5):
            est_t += weights[j+4] * td[idx+j, :]
            est_n += weights[j+4] * nd[idx+j, :]
            est_b += weights[j+4] * bd[idx+j, :]
        est_t *= 1./dphi
        est_n *= 1./dphi
        est_b *= 1./dphi
        assert np.all(np.abs(est_t - tdd[idx]) < 1e-6)
        assert np.all(np.abs(est_n - ndd[idx]) < 1e-6)
        assert np.all(np.abs(est_b - bdd[idx]) < 1e-6)
