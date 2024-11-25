For the following test case ttnn.reshape with the tensor in TILE_LAYOUT  is failing in lenet model.

To recreate the issue run the command:
`pytest tests/ttnn/unit_tests/operations/test_reshape_mesh.py::test_reshape_case_1`
