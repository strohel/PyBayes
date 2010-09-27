#!/bin/bash

# configure these:

# directory containing PyBayes's setup.py
PYBAYES_SRC_DIR="${HOME}/projekty/pybayes"

# directory where PyBayes is installed into (WILL BE DELETED during init)
PYBAYES_INSTALL_DIR="/usr/local/lib/python2.6/site-packages/pybayes"

# additionally, following env variables affect complete test:
# SETUP_ARGS, TEST_ARGS, STRESS_DATA_DIR, STRESS_ARGS

install_pybayes() {
	# TODO: dangerous!
	rm -rf "${PYBAYES_INSTALL_DIR}"

	pushd "${PYBAYES_SRC_DIR}" >/dev/null
	echo "Running \`./setup.py ${SETUP_ARGS} $@ install\` in ${PYBAYES_SRC_DIR}"
	./setup.py ${SETUP_ARGS} $@ install
	popd >/dev/null
}

for use_cython in no yes; do
	install_pybayes --use-cython=$use_cython
	python "${PYBAYES_SRC_DIR}/examples/run_tests.py" ${TEST_ARGS}
	python "${PYBAYES_SRC_DIR}/examples/run_stresses.py" --datadir "${STRESS_DATA_DIR:-${PYBAYES_SRC_DIR}/examples/stress_data}" ${STRESS_ARGS}
	echo
done
