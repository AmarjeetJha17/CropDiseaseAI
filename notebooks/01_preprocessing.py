import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ====================== CONFIG ======================
root = Path(".")
raw_dir = root / "data" / "raw"
processed_dir = root / "data" / "processed"
processed_dir.mkdir(exist_ok=True)

print("🔄 Starting improved preprocessing with diagnostics...\n")

# ====================== COLLECT IMAGES + DIAGNOSTICS ======================
data = []
class_counts = {}

for crop in ["soybean", "sugarcane"]:
    crop_dir = raw_dir / crop
    if not crop_dir.exists():
        print(f"⚠️ Folder missing: {crop_dir}")
        continue
    
    print(f"📁 Scanning {crop.upper()} folder...")
    
    for class_name in sorted(os.listdir(crop_dir)):
        class_dir = crop_dir / class_name
        if not class_dir.is_dir():
            continue
        
        # Clean class name (remove numbers, dots, extra spaces, underscores)
        clean_class = class_name.strip()
        clean_class = ''.join(c for c in clean_class if not c.isdigit())  # remove numbers
        clean_class = clean_class.replace('.', '').replace('_', ' ').strip()
        clean_class = ' '.join(clean_class.split())  # remove extra spaces
        clean_class = clean_class.title().replace(' ', '_')
        
        prefixed_class = f"{crop}_{clean_class}"
        
        # Count valid images
        image_count = 0
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = class_dir / img_name
                data.append({
                    'image_path': str(img_path.absolute()),
                    'label': prefixed_class,
                    'original_class': class_name
                })
                image_count += 1
        
        class_counts[prefixed_class] = image_count
        print(f"   → {prefixed_class}: {image_count:,} images")

print(f"\n✅ Total valid images found: {len(data):,}")
print(f"✅ Unique classes: {len(class_counts)}")

# ====================== FILTER SMALL CLASSES ======================
# Skip any class with < 30 images (too small for good training)
min_images = 30
filtered_data = [item for item in data if class_counts[item['label']] >= min_images]

df = pd.DataFrame(filtered_data)
print(f"✅ After filtering small classes: {len(df):,} images ({len(df['label'].unique())} classes)")

# Save full dataset
df.to_csv(processed_dir / "full_dataset.csv", index=False)

# ====================== STRATIFIED SPLIT ======================
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

train_df.to_csv(processed_dir / "train.csv", index=False)
val_df.to_csv(processed_dir / "val.csv", index=False)
test_df.to_csv(processed_dir / "test.csv", index=False)

print("\n📊 Final Split:")
print(f"   Train: {len(train_df):,} images")
print(f"   Val:   {len(val_df):,} images")
print(f"   Test:  {len(test_df):,} images")

print("\n📋 Final Class Distribution:")
print(df['label'].value_counts().sort_values(ascending=False))

print(f"\n🎉 Preprocessing complete! Files saved in: {processed_dir}")