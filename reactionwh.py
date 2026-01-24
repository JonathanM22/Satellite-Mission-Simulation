import numpy as np


class ReactionWh:

    def __init__(self, mass, J_spin, J_perp, wl_unit):
        self.mass = mass
        self.J_spin = J_spin
        self.J_perp = J_perp
        self.wl_unit = wl_unit
        self.wl = 0

    # Inertia of non-spin axises of reachtion wheel
    def Jwh_body_perp(self):
        return self.J_perp*(np.identity(3) - np.outer(self.wl_unit, self.wl_unit))

    # Hwh_body = angular momentum of wheels in body frame
    def Hwh_body(self, sat_w):
        return self.J_spin*(self.wl_unit*sat_w + self.wl)*self.wl_unit
