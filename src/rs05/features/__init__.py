import json
from ..data import artifact
def groups():return json.loads(artifact("protocol/primary",verify=True).read_text())["feature_groups"]
def specifications():return json.loads(artifact("protocol/primary").read_text())["spec_group_mapping"]
