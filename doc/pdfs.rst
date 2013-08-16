=============================
Probability Density Functions
=============================

.. automodule:: pybayes.pdfs
   :no-members:

Random Variables and their Components
=====================================

.. autoclass:: RV

   .. automethod:: __init__

.. autoclass:: RVComp

   .. automethod:: __init__

Probability Density Function prototype
======================================

.. autoclass:: CPdf

.. autoclass:: Pdf

Unconditional Probability Density Functions (pdfs)
==================================================

.. autoclass:: UniPdf

   .. automethod:: __init__

.. autoclass:: AbstractGaussPdf

.. autoclass:: GaussPdf

   .. automethod:: __init__

.. autoclass:: LogNormPdf

   .. automethod:: __init__

.. autoclass:: TruncatedNormPdf

   .. automethod:: __init__

.. autoclass:: GammaPdf

   .. automethod:: __init__

.. autoclass:: InverseGammaPdf

   .. automethod:: __init__

.. autoclass:: AbstractEmpPdf

.. autoclass:: EmpPdf

   .. automethod:: __init__

.. autoclass:: MarginalizedEmpPdf

   .. automethod:: __init__

.. autoclass:: ProdPdf

   .. automethod:: __init__

Conditional Probability Density Functions (cpdfs)
=================================================

In this section, variable :math:`c` in math exressions denotes condition.

.. autoclass:: MLinGaussCPdf

   .. automethod:: __init__

.. autoclass:: LinGaussCPdf

   .. automethod:: __init__

.. autoclass:: GaussCPdf

   .. automethod:: __init__

.. autoclass:: GammaCPdf

   .. automethod:: __init__

.. autoclass:: InverseGammaCPdf

   .. automethod:: __init__

.. autoclass:: ProdCPdf

   .. automethod:: __init__
