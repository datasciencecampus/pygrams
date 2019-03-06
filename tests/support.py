def assert_list_almost_equal(test, list_a, list_b, places=7):
    test.assertEqual(len(list_a), len(list_b), 'Lists must be same length')
    for a, b in zip(list_a, list_b):
        test.assertAlmostEqual(a, b, places=places)
