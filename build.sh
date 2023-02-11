rm -rf build dist
python3 setup.py sdist
twine upload dist/*
