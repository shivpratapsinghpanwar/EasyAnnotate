import os
import xml.etree.ElementTree as ET
from collections import Counter

# ==========================
# CONFIG
# ==========================
folder_path = r"D:\Data_factory\Projects\Project_Fight_NonFight\Fight_person_Data_pascal_voc"
# ==========================
print("starting: 1")
label_counter = Counter()  
corrupted_xmls = []

total_xmls = 0
total_objects = 0

for root, _, files in os.walk(folder_path):
    for file in files:
        if not file.lower().endswith(".xml"):
            continue

        xml_path = os.path.join(root, file)
        total_xmls += 1

        try:
            tree = ET.parse(xml_path)
            root_xml = tree.getroot()

            for obj in root_xml.findall("object"):
                name = obj.find("name")
                if name is not None:
                    label = name.text.strip()
                    label_counter[label] += 1
                    total_objects += 1

        except Exception as e:
            corrupted_xmls.append((xml_path, str(e)))

print("=" * 60)
print(f"XML Files Scanned : {total_xmls}")
print(f"Objects Found     : {total_objects}")
print("=" * 60)

print("\nLabels Found:\n")

for label, count in sorted(label_counter.items()):
    print(f"{label:<25} {count}")

print("\nUnique Labels:", len(label_counter))

if corrupted_xmls:
    print("\nCorrupted XML Files:")
    for path, err in corrupted_xmls:
        print(f"- {path}")
        print(f"  Error: {err}")

print("\nDone.")