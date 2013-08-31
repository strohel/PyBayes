==================
PyBayes Change Log
==================

This file mentions changes between PyBayes versions that are important for its users. Most
recent versions are mentioned on top.

Changes between 0.3 and 0.4
===========================

* `Filter.posterior`'s signature was changed to return `CPdf` instead of `Pdf`. It is more
  versatile - it for example allows you you to return zero-condition `ProdCPdf`.
* The optimised Cython PyBayes version was turned to use Cython memoryviews (with help
  from Ceygen_) instead of NumPy arrays where it makes sense. Cython memoryviews and NumPy
  arrays are mostly mutually compatible, but you may need to use :func:`np.asarray(...)
  <numpy.asarray>` or convert your code to use e.g. :func:`np.sum(pybayes_result)
  <numpy.sum>` instead of :obj:`pybayes_result.sum() <numpy.ndarray.sum>`. Alternatively,
  you can convert your code to use Cython memoryviews too.
* Use of bundled Tokyo is replaced by the Ceygen_ project and Tokyo submodule is removed.
* ParticleFilter.bayes() now ignores cond completely. Yell if you need it.
* ParticleFilter lost last emp_pdf argument. Pass the same object as the init_pdf argument
  to achieve the same thing.
* Test- and stress-suite no longer need PyBayes to be installed. (no privilege problems etc.)
* Build-system was rewritten so that it is no longer an ugly hack. .pxd and .py files are now
  installed along .so (.dll) files for interoperability and additional openness. Better parsing of
  setup.py arguments and custom parameters visible in the --help command.
* (C)Pdf shape() and cond_shape() functions are no longer abstract and just return
  `self.rv.dimension` and `self.cond_rv.dimension` respectively. CPdf subclasses therefore should
  not implement these methods. This is a backwards compatible change API-wise.

.. _Ceygen: https://github.com/strohel/Ceygen
