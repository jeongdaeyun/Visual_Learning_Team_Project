import os
import json

json_folder = 'please check your result folder(no json file)'

total = 0
correct = 0

json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

for json_file in json_files:
    json_path = os.path.join(json_folder, json_file)

    with open(json_path, 'r') as f:
        data = json.load(f)
        
        for item in data:
            total += 1
            if item['pos_caption'] == item['predict_caption']:
                correct += 1
    
accuracy = correct / total if total > 0 else 0

print(f"Total samples: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2%}")
