#!/usr/bin/env bash
rm dist/*
rm -R prismx_lachmann12*
cp setup_template.py setup.py

sed -i "s/versionNumber/$1/g" setup.py

key=`cat key.txt`

python3 setup.py sdist bdist_wheel
python3 -m twine upload -u __token__ -p $key --repository testpypi dist/*
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps prismx-lachmann12