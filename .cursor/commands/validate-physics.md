# validate-physics


Run pytest specifically on tests/test_astro_physics.py and tests/test_validators.py. Ensure all Pydantic boundaries (like T_eq constraints and mass-radius consistency) are passing. Report any failures immediately without attempting to fix them first.