import json

def calculate_accuracy(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    total = len(data)
    correct = sum(1 for item in data if item["true_caption"] == item["predict_caption"])
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

json_path = "/aro_json_result_path"

accuracy, correct, total = calculate_accuracy(json_path)

print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy * 100:.2f}%")