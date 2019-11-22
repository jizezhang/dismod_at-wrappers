import sys
from cascade_at.settings import settings
from cascade_at.collectors.measurement_inputs import MeasurementInputsFromSettings

def get_input_from_db(model_version_id: int):
    sts = settings.settings_from_model_version_id(model_version_id, "dismod-at-dev")
    inputs = MeasurementInputsFromSettings(sts)
    inputs.configure_inputs_for_dismod()

    df = inputs.dismod_data
    return df



