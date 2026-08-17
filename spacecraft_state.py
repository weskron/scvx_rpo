class SpacecraftState:
    """
    The idea here is to create an object that can store all the information about the spacecraft and then can translate between frames
    """
    def __init__(self):
        # position ECI

        # velocity ECI

        # mass

        # quaternion

        # intertial

        # keplerian orbital elements

    # TODO: Update to match the format of the class
    def keplerian_to_cartesian(self, a , e, inc, RAAN, aop, anomaly, mu, frame="inert", anomaly_type="mean"):
    """
    Convert from Keplerian orbital elements to cartesian coordinates in the inertial frame. 
    """
    # Step 1: Find True Anomaly (nu) and Eccentric Anomaly (E)
    if anomaly_type == "mean":
        E = solve_kepler(anomaly, e)
        nu = eccentric_to_true(E, e)
    elif anomaly_type == "true":
        nu = anomaly
        E = true_to_eccentric(anomaly, e) # do the reverse
    else:
        raise ValueError("anomaly_type must be 'mean' or 'true'")
    

    # Step 2: Find orbital radius
    r_c = a * (1 - e * math.cos(E))

    # Step 3: Find the position and velocity vector in the orbital frame
    pos_o = r_c * np.array([math.cos(nu), math.sin(nu), 0])
    vel_o = (math.sqrt(mu / a) / (1 - e * math.cos(E))) * np.array([-math.sin(E), math.sqrt(1 - e**2) * math.cos(E), 0])

    # Step 4: Transform the position and velocity vector to the intertial frame.
    if frame == "inert": 
        NO_dcm = RTN_to_inert_313(RAAN, inc, aop)
        pos_i = NO_dcm @ pos_o
        vel_i = NO_dcm @ vel_o
    else:
        pos_i = pos_o
        vel_i = vel_o

    return pos_i, vel_i
