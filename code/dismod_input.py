import sys
from cascade_at.settings import settings
from cascade_at.collector.measurement_inputs import MeasurementInputsFromSettings
from cascade_at.settings.base_case import BASE_CASE

def get_settings(model_version_id=None, settings_json=None):
    if model_version_id is not None:
        return settings.settings_from_model_version_id(model_version_id, "dismod-at-dev")
    elif settings_json is not None:
        return settings.load_settings(settings_json)
    else:
        return settings.load_settings(BASE_CASE)

def get_input_from_db(model_version_id=None, settings_json=None):
    #sts = settings.settings_from_model_version_id(model_version_id, "dismod-at-dev")
    sts = get_settings(model_version_id, settings_json)
    inputs = MeasurementInputsFromSettings(sts)
    inputs.configure_inputs_for_dismod(sts)

    df = inputs.dismod_data
    return df
