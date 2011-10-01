==================
PyBayes Change Log
==================

This file mentions important changes between PyBayes version that are important for its users. Most
recent versions are mentioned on top.

Changes between 0.3 and 0.4
===========================

* Test-suite no longer needs PyBayes to be installed, yay! (no privilege problems etc.)
* Build-system was rewritten so that it is no longer an ugly hack. .pxd and .py files are now
  installed along .so (.dll) files for interoperability and additional openness. Better parsing of
  setup.py arguments and custom parameters visible in the --help command.
* (C)Pdf shape() and cond_shape() functions are no longer abstract and just return
  `self.rv.dimension` and `self.cond_rv.dimension` respectively. CPdf subclasses therefore should
  not implement these methods. This is a backwards compatible change API-wise.
