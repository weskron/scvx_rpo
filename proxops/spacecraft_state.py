import numpy as np
from proxops.state_conversions import KeplerianElements, keplerian_to_eci, eci_to_keplerian
class SpacecraftState:
    """
    Base class for storing a spacecraft state. Can be initialized using an ECI state or keplerian state. Must include intertial properties
    """
    def __init__(self, quaternion, mass: float, width: float, height: float, length: float, keplerian = None, r_eci = None, v_eci = None,):
        if r_eci is not None and v_eci is not None:
            # position ECI
            self.r_eci = np.asarray(r_eci, dtype=float)
            # velocity ECI
            self.v_eci = np.asarray(v_eci, dtype=float)
        else:
            self.r_eci, self.v_eci = self.from_keplerian()
        # keplerian orbital elements
        self.keplerian = keplerian if keplerian is not None else self.from_ECI()
        # mass
        self.mass = mass
        # quaternion scalar last 
        self.quaternion = np.asarray(quaternion, dtype=float)
        # intertial
        self.intertia = (1/12) * self.mass * np.diag([
                            width**2 + height**2,
                            length**2 + height**2,
                            length**2 + width**2
                        ])
        # validate entries
        self.validate(length, width, height)

    def from_keplerian(self):
        # r_eci, v_eci = keplerian_to_eci(keplerian)  # A2
        r_eci = [1, 0, 0]
        v_eci = [1, 0, 0]
        return r_eci, v_eci

    def from_ECI(self):
        # keplerian = eci_to_keplerian(r_eci, v_eci) # A2 update
        keplerian = KeplerianElements(semi_major_axis = 1, eccentricity = 0, inclination = 0, raan = 0, argp=0, nu=0)
        return keplerian

    def orbital_radius(self):
        return np.linalg.norm(self.r_eci)

    def specific_angular_momentum_vector(self):
        return np.linalg.cross(self.r_eci, self.v_eci)


    def validate(self, length, width, height):
        # length, width, height must be greater than 0. 
        if length < 0 or width < 0 or height < 0:
            raise ValueError("rectangle dimensions must be greater than or equal to 0")

        # norm of quaternion must be 1
        if abs(np.linalg.norm(self.quaternion) - 1) > 1e-8:
            raise ValueError("The quaternion norm must be equal to 1")

        # mass must be greater than 0
        if self.mass <= 0:
            raise ValueError("mass must be greater than 0")
        
        # Add r_eci and v_eci check!?!?!
        if self.orbital_radius() == 0:
            raise ValueError("Orbital radius must be greater than 0")