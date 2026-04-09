"""
ml_pipeline.use_cases
=====================
Use-case-specific pipeline implementations.

Modules
-------
fraud_detection : Telco online-payment fraud feature engineering, constants,
                  and model configuration.
fraud_pipeline  : Concrete FraudDetectionPipeline that wires everything together.
"""

from .fraud_pipeline import FraudDetectionPipeline  # noqa: F401
