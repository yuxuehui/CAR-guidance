from ..core.base_experiment import BaseExperiment
from ..guidance.none_guidance import NoneGuidance

class BaselineExperiment(BaseExperiment):

    def _create_guidance(self):
        return NoneGuidance()
