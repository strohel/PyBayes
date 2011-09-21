==================
PyBayes Change Log
==================

This file mentions important changes between PyBayes version that are important for its users. Most
recent versions are mentioned on top.

Changes between 0.3 and 0.4
===========================

* (C)Pdf shape() and cond_shape() functions are no longer abstract and just return
  `self.rv.dimension` and `self.cond_rv.dimension` respectively. CPdf subclasses therefore should
  not implement these methods. This is a backwards compatible change API-wise.
