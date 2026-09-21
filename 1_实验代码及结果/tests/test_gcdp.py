from mprbps.verification import verify_gcdp


def test_gcdp_matches_exhaustive():
    verify_gcdp(trials=5, seed=3)
