from build_external_mixed_dataset import domain_split_ids


def test_domain_split_ids_are_disjoint_and_deterministic():
    first = domain_split_ids(
        list(range(13)),
        validation_images=2,
        test_images=3,
        seed=7,
    )
    second = domain_split_ids(
        list(range(13)),
        validation_images=2,
        test_images=3,
        seed=7,
    )
    assert first == second
    train, validation, test = first
    assert len(train) == 8
    assert len(validation) == 2
    assert len(test) == 3
    assert not (set(train) & set(validation))
    assert not (set(train) & set(test))
    assert not (set(validation) & set(test))
    assert set(train) | set(validation) | set(test) == set(range(13))
