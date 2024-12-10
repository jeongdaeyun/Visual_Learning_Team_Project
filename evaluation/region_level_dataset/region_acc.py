import json

def calculate_accuracy(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    total = len(data)  
    correct = 0  

    for entry in data:
        if entry["gt_caption"] == entry["predict_phrase"]:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

json_path_att = "/your_attribute_json_result"
json_path_obj = "/your_object_json_result"
json_path_rel = "/your_relation_json_result"

accuracy_att = calculate_accuracy(json_path_att)
accuracy_obj = calculate_accuracy(json_path_obj)
accuracy_rel = calculate_accuracy(json_path_rel)

print(f"obj Accuracy: {accuracy_obj:.2%}")
print(f"rel Accuracy: {accuracy_rel:.2%}")
print(f"att Accuracy: {accuracy_att:.2%}")

