import numpy as np
import jax.numpy as jnp
from jax import vjp

from .jit import jit
from .framedcurve import FramedCurve, FrameRotation, ZeroRotation, FramedCurveCentroid, FramedCurveFrenet, inner

"""
The functions and classes in this model are used to deal with multifilament
approximation of finite build coils.
"""

__all__ = ['create_multifilament_grid', 'CurveFilament']


class CurveFilament(FramedCurve):

    def __init__(self, framedcurve, dn, db):
        """
        Given a FramedCurve, defining a normal and
        binormal vector, create a grid of curves by shifting 
        along the normal and binormal vector. 

        The idea is explained well in Figure 1 in the reference:

        Singh et al, "Optimization of finite-build stellarator coils",
        Journal of Plasma Physics 86 (2020),
        doi:10.1017/S0022377820000756. 

        Args:
            curve: the underlying curve
            dn: how far to move in normal direction
            db: how far to move in binormal direction
            rotation: angle along the curve to rotate the frame.
        """
        self.curve = framedcurve.curve 
        self.dn = dn
        self.db = db
        self.rotation = framedcurve.rotation 
        # self.rotation = FilamentRotation(framedcurve.rotation, )
        self.framedcurve = framedcurve 
        # deps = [self.framedcurve, self.rotation]
        FramedCurve.__init__(self, self.curve, self.rotation)

        # self.torsion = jit(
        #     lambda gamma_c, gamma_f, gammadash_c, gammadash_f, gammadashdash_f, alpha_c, alphadash_c: torsion_pure_centroid(
        #         gamma_c, gamma_f, gammadash_c, gammadash_f, gammadashdash_f, alpha_c, alphadash_c
        #     ) 
        # )
        self.torsion = jit(
            lambda b, ndash, gammadash: torsion_pure(
                b, ndash, gammadash
            )
        )
        self.torsiongrad_vjp0 = jit(lambda b, ndash, gammadash, v: vjp(
                lambda g: self.torsion(g, ndash, gammadash), b
            )[1](v)[0]
        )
        self.torsiongrad_vjp1 = jit(lambda b, ndash, gammadash, v: vjp(
                lambda g: self.torsion(b, g, gammadash), ndash
            )[1](v)[0]
        )
        self.torsiongrad_vjp2 = jit(lambda b, ndash, gammadash, v: vjp(
                lambda g: self.torsion(b, ndash, g), gammadash
            )[1](v)[0]
        )
        self.binorm = jit(
            lambda b, tdash, gammadash: binormal_curvature_pure(
                b, tdash, gammadash
            )
        )
        self.binormgrad_vjp0 = jit(lambda b, tdash, gammadash, v: vjp(
                lambda g: self.binorm(g, tdash, gammadash), b
            )[1](v)[0]
        )
        self.binormgrad_vjp1 = jit(lambda b, tdash, gammadash, v: vjp(
                lambda g: self.binorm(b, g, gammadash), tdash
            )[1](v)[0]
        )
        self.binormgrad_vjp2 = jit(lambda b, tdash, gammadash, v: vjp(
                lambda g: self.binorm(b, tdash, g), gammadash
            )[1](v)[0]
        )

    def frame_torsion(self):
        """
        Returns the frame torsion along the CurveFilament
        """
        _, _, b = self.framedcurve.rotated_frame()
        _, ndash, _ = self.framedcurve.rotated_frame_dash()
        return self.torsion(b, ndash, self.gammadash())

    def dframe_torsion_by_dcoeff_vjp(self, v):
        """
        VJP function for derivatives of the frame torsion with respect to the
        curve and rotation dofs.
        """
        _, _, b = self.framedcurve.rotated_frame()
        _, ndash, _ = self.framedcurve.rotated_frame_dash()
        gammadash = self.gammadash()

        grad0 = self.torsiongrad_vjp0(b, ndash, gammadash, v)
        grad1 = self.torsiongrad_vjp1(b, ndash, gammadash, v)
        grad2 = self.torsiongrad_vjp2(b, ndash, gammadash, v)

        return self.db_by_dcoeff_vjp(grad0) \
            + self.dndash_by_dcoeff_vjp(grad1) \
            + self.dgammadash_by_dcoeff_vjp(grad2)

    def db_by_dcoeff_vjp(self, v):
        return self.framedcurve.rotated_frame_dcoeff_vjp(
            np.zeros_like(v), np.zeros_like(v), v
        )

    def dndash_by_dcoeff_vjp(self, v):
        return self.framedcurve.rotated_frame_dash_dcoeff_vjp(
            np.zeros_like(v), v, np.zeros_like(v)
        )

    def frame_binormal_curvature(self):
        """
        Returns the frame binormal curvature along the CurveFilament
        """
        _, _, b = self.framedcurve.rotated_frame()
        tdash, _, _ = self.framedcurve.rotated_frame_dash()
        return self.binorm(b, tdash, self.gammadash())

    def dframe_binormal_curvature_by_dcoeff_vjp(self, v):
        """
        VJP function for derivatives of the frame binormal curvature with 
        respect to the curve and rotation dofs.
        """
        _, _, b = self.framedcurve.rotated_frame()
        tdash, _, _ = self.framedcurve.rotated_frame_dash()
        gammadash = self.gammadash()  

        grad0 = self.binormgrad_vjp0(b, tdash, gammadash, v)
        grad1 = self.binormgrad_vjp1(b, tdash, gammadash, v)
        grad2 = self.binormgrad_vjp2(b, tdash, gammadash, v)

        return self.db_by_dcoeff_vjp(grad0) \
            + self.dtdash_by_dcoeff_vjp(grad1) \
            + self.dgammadash_by_dcoeff_vjp(grad2)

    def dtdash_by_dcoeff_vjp(self, v):
        return self.framedcurve.rotated_frame_dash_dcoeff_vjp(
            v, np.zeros_like(v), np.zeros_like(v)
        )

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma(self):
        t, n, b = self.framedcurve.rotated_frame()
        return self.curve.gamma() + self.dn * n + self.db * b

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        t, n, b = self.framedcurve.rotated_frame()
        gamma[:] = self.curve.gamma() + self.dn * n + self.db * b

    def gammadash(self):
        td, nd, bd = self.framedcurve.rotated_frame_dash()
        return self.curve.gammadash() + self.dn * nd + self.db * bd

    def gammadash_impl(self, gammadash):
        td, nd, bd = self.framedcurve.rotated_frame_dash()
        gammadash[:] = self.curve.gammadash() + self.dn * nd + self.db * bd

    def gammadashdash_impl(self, gammadashdash):
        """
        Implementation of gammadashdash once I know the mathematical formula
        """
        tdd, ndd, bdd = self.framedcurve.rotated_frame_dashdash()
        gammadashdash[:] = self.curve.gammadashdash() + self.dn * ndd + self.db * bdd

    def dgamma_by_dcoeff_vjp(self, v):
        return self.curve.dgamma_by_dcoeff_vjp(v) \
           +  self.framedcurve.rotated_frame_dcoeff_vjp(np.zeros_like(v), self.dn*v, self.db*v)

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v) \
           +  self.framedcurve.rotated_frame_dash_dcoeff_vjp(np.zeros_like(v), self.dn*v, self.db*v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        """
        Implementation of gammadashdash_by_dcoeff_vjp once I have the jvp functions
        defined for rotated frame dash (or is it another object)
        """
        return self.curve.dgammadashdash_by_dcoeff_vjp(v) \
           +  self.framedcurve.rotated_frame_dashdash_dcoeff_vjp(np.zeros_like(v), self.dn*v, self.db*v)

# def torsion_pure_centroid(gamma_c, gamma_f, gammadash_c, gammadash_f,
#                           gammadashdash_f, alpha_c, alphadash_c):
#     _, _, b = rotated_centroid_frame(gamma, gammadash, alpha)
#     _, ndash, _ = rotated_centroid_frame_dash(
#         gamma, gammadash, gammadashdash, alpha, alphadash)

#     ndash *= 1/jnp.linalg.norm(gammadash, axis=1)[:, None]
#     return inner(ndash, b)

def torsion_pure(b, ndash, gammadash):
    """
    b and ndash come from centerline curve, gammadash comes from offset curve.
    """
    ndash *= 1/jnp.linalg.norm(gammadash, axis=1)[:, None]
    return inner(ndash, b)

def binormal_curvature_pure(b, tdash, gammadash):
    """
    b and tdash come from centerline curve, gammadash comes from offset curve.
    """
    tdash *= 1/jnp.linalg.norm(gammadash, axis=1)[:, None]
    return inner(tdash, b)


class FilamentRotation(FrameRotation):
    def __init__(self, curve_filament):
        """
        Defines a rotation angle needed to align finite build curve filaments
        with the rotation of the center-line curve. 
        This is necessary if one wants to strain optimize a finite build coil.
        Creating a FramedCurve from a CurveFilament will rely on the
        rotated frame of the centerline coil that the CurveFilament was defined
        by. We want the same frame, but the rotation angle for a CurveFilament
        is needed to compute that frame is different. This rotation computes
        the rotation considering this effect.

        alpha_f = alpha_0 + cos^-1(n_0^hat \cdot n_f^hat)
        """        
        self.curve_filament = curve_filament
        self.center_curve = self.curve_filament.framedcurve
        self.center_rotation = self.curve_filament.rotation
        super().__init__(
            quadpoints=self.curve_filament.quadpoints,
            order=self.center_rotation.order,
            scale=self.center_rotation.scale
        )
    
    def alpha(self, quadpoints):
        """
        Issues:
        - If generated without dn or db, causes nan errors with floating point precision on arccos
        - Build in dependence on quadpoints for easy derivatives 
        """
        # Get unrotated frame of centerline coil
        _, self.n_0, self.b_0 = self.center_curve.unrotated_frame()
        # Get unrotated frame of CurveFilament by creating a FramedCurveCentroid
        # to have access to unrotated_frame function
        fc = FramedCurveCentroid(self.curve_filament) # this should be probably for both frenet/centroid
        _, self.n_f, self.b_f = fc.unrotated_frame()
        # get the alpha based on the centerlines quadpoints
        self.alpha_0 = self.center_rotation.alpha(quadpoints)
        #### DEBUGGING
        self.to_rotate_by = jnp.arccos(
            # performs row-wise dot products
            jnp.einsum('ik,ik->i', self.n_0, self.n_f) 
        )
        #####
        self.alpha_f = self.alpha_0 + jnp.arccos(
            # performs row-wise dot products
            jnp.einsum('ik,ik->i', self.n_0, self.n_f) 
        )
        return self.alpha_f


def create_multifilament_grid(curve, numfilaments_n, numfilaments_b, gapsize_n, gapsize_b, 
                              rotation_order=None, rotation_scaling=None, frame='centroid'):
    """
    Create a regular grid of ``numfilaments_n * numfilaments_b`` many
    filaments to approximate a finite-build coil.

    Note that "normal" and "binormal" in the function arguments here
    refer to either the Frenet frame or the "coil centroid
    frame" defined by Singh et al., before rotation.

    Args:
        curve: The underlying curve.
        numfilaments_n: number of filaments in normal direction.
        numfilaments_b: number of filaments in bi-normal direction.
        gapsize_n: gap between filaments in normal direction.
        gapsize_b: gap between filaments in bi-normal direction.
        rotation_order: Fourier order (maximum mode number) to use in the expression for the rotation
                        of the filament pack. ``None`` means that the rotation is not optimized.
        rotation_scaling: scaling for the rotation degrees of freedom. good
                           scaling improves the convergence of first order optimization
                           algorithms. If ``None``, then the default of ``1 / max(gapsize_n, gapsize_b)``
                           is used.
        frame: orthonormal frame to define normal and binormal before rotation (either 'centroid' or 'frenet')
    """
    assert frame in ['centroid', 'frenet']
    if numfilaments_n % 2 == 1:
        shifts_n = np.arange(numfilaments_n) - numfilaments_n//2
    else:
        shifts_n = np.arange(numfilaments_n) - numfilaments_n/2 + 0.5
    shifts_n = shifts_n * gapsize_n
    if numfilaments_b % 2 == 1:
        shifts_b = np.arange(numfilaments_b) - numfilaments_b//2
    else:
        shifts_b = np.arange(numfilaments_b) - numfilaments_b/2 + 0.5
    shifts_b = shifts_b * gapsize_b

    if rotation_scaling is None:
        rotation_scaling = 1/max(gapsize_n, gapsize_b)
    if rotation_order is None:
        rotation = ZeroRotation(curve.quadpoints)
    else:
        rotation = FrameRotation(curve.quadpoints, rotation_order, scale=rotation_scaling)
    if frame == 'frenet':
        framedcurve = FramedCurveFrenet(curve, rotation)
    else:
        framedcurve = FramedCurveCentroid(curve, rotation)

    filaments = []
    for i in range(numfilaments_n):
        for j in range(numfilaments_b):
            filaments.append(CurveFilament(framedcurve, shifts_n[i], shifts_b[j]))
    return filaments

