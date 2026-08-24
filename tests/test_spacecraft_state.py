from proxops.spacecraft_state import SpacecraftState
from proxops.state_conversions import KeplerianElements
import pytest 

# test generic construction of spacecraft state
def test_spacecraft_state():
    eci_state_test = SpacecraftState(
        quaternion = [0, 0, 0, 1],
        mass = 1,
        width = 1,
        height = 1,
        length = 1,
        r_eci = [1, 1, 1],
        v_eci = [1, 1, 1]
    )
    assert eci_state_test.keplerian != None

    # build keplerian state for test
    keplerian = KeplerianElements(
        semi_major_axis=10_000,
        eccentricity = 0,
        inclination = 0,
        raan = 0,
        argp = 0,
        nu = 0,
    )

    kep_state_test = SpacecraftState(
        mass = 1,
        quaternion = [0, 0, 0, 1],
        width = 1,
        height = 1,
        length = 1,
        keplerian = keplerian
    )
    assert kep_state_test.r_eci != None
    assert kep_state_test.v_eci != None

    return None

def test_missing_inertia():
    # Test Inertia creation
    keplerian = KeplerianElements(
        semi_major_axis=10_000,
        eccentricity = 0,
        inclination = 0,
        raan = 0,
        argp = 0,
        nu = 0,
    )

    with pytest.raises(ValueError, match="rectangle dimensions must be greater than or equal to 0"):
        bad_inertia = SpacecraftState(
            mass = 1,
            quaternion = [0, 0, 0, 1],
            width = -1,
            height = 0,
            length = 0,
            keplerian = keplerian
        )
    
    # Raise error if a required entry is not given. 
    with pytest.raises(TypeError):
        bad_inertia = SpacecraftState(
            mass = 1,
            quaternion = [0, 0, 0, 1],
            width = -1,
            keplerian = keplerian
        )
    
    return None
