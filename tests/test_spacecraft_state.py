from proxops.state_conversions import (
    KeplerianElements, keplerian_to_eci, eci_to_keplerian
)
import pytest

EARTH_MU = 398600.4418  # km^3/s^2


# test that constructing keplerian state works
def test_keplerian_state():
    keplerian = KeplerianElements(
        semi_major_axis=10_000,   # km
        eccentricity=0,
        inclination=0,
        raan=0,
        argp=0,
        nu=0,
    )
    assert keplerian.semi_major_axis == 10_000
    assert keplerian.eccentricity == 0


# test no anomaly given
def test_no_anomaly():
    with pytest.raises(ValueError, match="Please enter mean anomaly value"):
        KeplerianElements(
            semi_major_axis=10_000,
            eccentricity=0,
            inclination=0,
            raan=0,
            argp=0
        )


# test bad semi major axis
def test_bad_semi_major_axis():
    with pytest.raises(ValueError, match=f"Semi major axis must be positive, entry was {-10_000}"):
        KeplerianElements(
            semi_major_axis=-10_000,
            eccentricity=0,
            inclination=0,
            raan=0,
            argp=0,
            nu=0,
        )


# test that keplerian -> eci -> keplerian recovers the original elements
def test_keplerian_eci_roundtrip():
    original = KeplerianElements(
        semi_major_axis=6871,     # km, ~500 km altitude
        eccentricity=0.01,
        inclination=0.5,           # nonzero: avoid the equatorial degenerate case
        raan=0.3,
        argp=0.2,
        nu=0.1,
    )

    r_eci, v_eci = keplerian_to_eci(original, mu=EARTH_MU)
    recovered = eci_to_keplerian(r_eci, v_eci, mu=EARTH_MU)

    assert recovered.semi_major_axis == pytest.approx(original.semi_major_axis, rel=1e-6)
    assert recovered.eccentricity == pytest.approx(original.eccentricity, abs=1e-8)
    assert recovered.inclination == pytest.approx(original.inclination, abs=1e-8)
    assert recovered.raan == pytest.approx(original.raan, abs=1e-8)
    assert recovered.argp == pytest.approx(original.argp, abs=1e-8)
    assert recovered.nu == pytest.approx(original.nu, abs=1e-8)