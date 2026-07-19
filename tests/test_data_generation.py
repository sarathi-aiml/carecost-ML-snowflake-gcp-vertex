from generate_claims import generate_claims, REQUIRED_COLUMNS


def test_columns_present():
    df = generate_claims(member_count=150, random_seed=42)
    assert list(df.columns) == REQUIRED_COLUMNS


def test_claim_ids_unique():
    df = generate_claims(member_count=150, random_seed=42)
    assert df["CLAIM_ID"].is_unique


def test_paid_amount_nonnegative():
    df = generate_claims(member_count=150, random_seed=42)
    assert (df["PAID_AMOUNT"] >= 0).all()


def test_service_date_populated():
    df = generate_claims(member_count=150, random_seed=42)
    assert df["SERVICE_DATE"].notna().all()


def test_deterministic():
    a = generate_claims(member_count=150, random_seed=42)
    b = generate_claims(member_count=150, random_seed=42)
    assert a.equals(b)
