check:
	mypy src/aeroespacial --ignore-missing-imports
format:
	pre-commit run --all-files