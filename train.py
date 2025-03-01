def get_feature_keys(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and "schedule" in data and "features" in data["schedule"]:
        return sorted(data["schedule"]["features"].keys())
    return []

def extract_features_from_json(json_path, feature_keys):
    with open(json_path, 'r') as f:
        data = json.load(f)
    sequence = []
    target = None
    if isinstance(data, dict):
        if "schedule" in data and "features" in data["schedule"]:
            features = data["schedule"]["features"]
            sequence.append([features.get(key, 0) for key in feature_keys])
        if "execution_time" in data:
            target = data["execution_time"]
    if not sequence or target is None:
        return None, None
    return sequence, target
