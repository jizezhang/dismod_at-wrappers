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

def get_input_from_db(model_version_id=None, settings_json=None, location_ids=None):
    sts = get_settings(model_version_id, settings_json)
    inputs = MeasurementInputsFromSettings(sts)
    inputs.get_raw_inputs()

    df = inputs.data.raw[['location_name', 'location_id', 'sex', 'year_start', 'year_end', 
                          'age_start', 'age_end', 'measure', 'mean', 'standard_error']]
    if location_ids is not None:
        df = df[df['location_id'].isin(location_ids)]
    
    return df
