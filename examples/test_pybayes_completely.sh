#!/bin/bash

# configure these:

# directory containing PyBayes's setup.py
PYBAYES_SRC_DIR="${HOME}/projekty/pybayes"

# directory where PyBayes is installed into (WILL BE DELETED during init)
PYBAYES_INSTALL_DIR="/usr/local/lib/python2.6/site-packages/pybayes"

# additionally, following env variables affect complete test:
# CLEAN_BEFORE_SETUP, SETUP_ARGS, TEST_ARGS, STRESS_DATA_DIR, STRESS_ARGS

install_pybayes() {
	if [[ -n "${PYBAYES_INSTALL_DIR}" && -n "${CLEAN_BEFORE_SETUP}" ]]; then
		echo "Running rm -rf \"${PYBAYES_INSTALL_DIR}\""
		rm -rf "${PYBAYES_INSTALL_DIR}"
	else
		echo "Not cleaning existing PyBayes install in \"${PYBAYES_INSTALL_DIR}\"."
		echo "This may result in mixed Python-Cython install!"
	fi

	pushd "${PYBAYES_SRC_DIR}" >/dev/null
	echo "Running \`./setup.py ${SETUP_ARGS} $@ clean\` in ${PYBAYES_SRC_DIR}"
	./setup.py ${SETUP_ARGS} $@ clean
	echo "Running \`./setup.py ${SETUP_ARGS} $@ install\` in ${PYBAYES_SRC_DIR}"
	./setup.py ${SETUP_ARGS} $@ install
	return=$?
	popd >/dev/null
	return $return
}

for use_cython in no yes; do
	if install_pybayes --use-cython=$use_cython; then
		echo "Running python \"${PYBAYES_SRC_DIR}/examples/run_tests.py\" ${TEST_ARGS}"
		python "${PYBAYES_SRC_DIR}/examples/run_tests.py" ${TEST_ARGS}

		echo "Running python \"${PYBAYES_SRC_DIR}/examples/run_stresses.py\" --datadir \"${STRESS_DATA_DIR:-${PYBAYES_SRC_DIR}/examples/stress_data}\" ${STRESS_ARGS}"
		python "${PYBAYES_SRC_DIR}/examples/run_stresses.py" --datadir "${STRESS_DATA_DIR:-${PYBAYES_SRC_DIR}/examples/stress_data}" ${STRESS_ARGS}
		echo
	else
		echo "Installing PyBayes failed"
	fi
done
