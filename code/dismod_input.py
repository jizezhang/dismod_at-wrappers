import sys
from cascade_at.settings import settings
from cascade_at.collector.measurement_inputs import MeasurementInputsFromSettings

def get_input_from_db(model_version_id=None, settings_json=None):
    sts = None
    if model_version_id is not None:
        sts = settings.settings_from_model_version_id(model_version_id, "dismod-at-dev")
    elif settings_json is not None:
        sts = settings.load_settings(settings_json)
    inputs = MeasurementInputsFromSettings(sts)
    inputs.configure_inputs_for_dismod(sts)

    df = inputs.dismod_data
    return df



