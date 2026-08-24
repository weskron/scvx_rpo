from proxops.state_conversions import KeplerianElements
import pytest

# test that constructing keplerian state works
def test_keplerian_state():
    keplerian = KeplerianElements(
        semi_major_axis=10_000,
        eccentricity = 0,
        inclination = 0,
        raan = 0,
        argp = 0,
        nu = 0,
    )
    return None

# test no anomaly given
def test_no_anomaly():
    with pytest.raises(ValueError, match="Please enter mean anomaly value"):
        KeplerianElements(
            semi_major_axis=10_000,
            eccentricity = 0,
            inclination = 0,
            raan = 0,
            argp = 0
        )
    return None


# test bad semi major axis
def test_bad_semi_major_axis():
    with pytest.raises(ValueError, match=f"Semi major axis must be positive, entry was {-10_000}"):
        KeplerianElements(
            semi_major_axis=-10_000,
            eccentricity = 0,
            inclination = 0,
            raan = 0,
            argp = 0,
            nu = 0,
        )
    return None


