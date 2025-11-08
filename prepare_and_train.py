"""
Prepare training data and train the Robot vs Human classifier.
This script will:
1. Copy your data from data_for_training to the expected data/raw structure
2. Split the data into train/val/test sets
3. Train the model
"""
import os
import shutil
try:
    from src import config
    from src.data_preprocessing import split_dataset, create_data_generators
except ImportError:
    # Fallback if running from inside src directory
    import config
    from data_preprocessing import split_dataset, create_data_generators
from src.train import train_model

def prepare_data(source_base=None):
    """Copy data from data_for_training to data/raw with correct structure.

    Args:
        source_base: Optional override path containing 'Human' and 'Robot(i)' folders.
    """
    print("=" * 50)
    print("PREPARING TRAINING DATA")
    print("=" * 50)

    # Determine base directory containing training folders
    if source_base is None:
        # default relative to project root
        project_root = os.path.dirname(os.path.abspath(__file__))
        source_base = os.path.join(project_root, 'data_for_training')

    # Accept both spellings for backward compatibility
    source_human = os.path.join(source_base, 'Human')
    # Original code had 'Roboti'; support both 'Robot' and 'Roboti'
    source_robot = os.path.join(source_base, 'Robot')
    if not os.path.isdir(source_robot):
        source_robot = os.path.join(source_base, 'Roboti')

    # Destination directories (what the model expects)
    dest_base = config.RAW_DATA_DIR
    dest_human = os.path.join(dest_base, 'human')
    dest_robot = os.path.join(dest_base, 'robot')
    os.makedirs(dest_human, exist_ok=True)
    os.makedirs(dest_robot, exist_ok=True)

    # Validate sources
    missing = []
    if not os.path.exists(source_human):
        missing.append(source_human)
    if not os.path.exists(source_robot):
        missing.append(source_robot)
    if missing:
        print("ERROR: Missing source data directories:")
        for m in missing:
            print(f" - {m}")
        print("Please create them or pass --source-base pointing to the correct path.")
        return False

    def _copy(src_dir, dst_dir, label):
        print(f"\nCopying {label} images from: {src_dir}")
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        print(f"Found {len(files)} {label} images")
        copied = 0
        for file in files:
            src = os.path.join(src_dir, file)
            dst = os.path.join(dst_dir, file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
        print(f"Copied {copied} new files to: {dst_dir}")
        return len(files)

    human_count = _copy(source_human, dest_human, 'human')
    robot_count = _copy(source_robot, dest_robot, 'robot')

    print(f"\n✅ Data preparation complete!")
    print(f"   Human images: {human_count}")
    print(f"   Robot images: {robot_count}")
    print(f"   Total: {human_count + robot_count} images")
    return True


def main():
    """
    Main function to prepare data and train model.
    """
    print("\n" + "=" * 70)
    print(" ROBOT VS HUMAN CLASSIFIER - TRAINING PIPELINE")
    print("=" * 70)

    # Allow optional environment variable to override source path
    source_base = os.environ.get('TRAINING_SOURCE_DIR')
    if not prepare_data(source_base):
        print("\n❌ Data preparation failed!")
        return

    # Step 2: Split data into train/val/test sets
    print("\n" + "=" * 50)
    print("STEP 2: SPLITTING DATA")
    print("=" * 50)
    split_dataset(
        config.RAW_DATA_DIR,
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.TEST_RATIO
    )

    # Step 3: Verify data generators work
    print("\n" + "=" * 50)
    print("STEP 3: CREATING DATA GENERATORS")
    print("=" * 50)
    train_gen, val_gen = create_data_generators(
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.BATCH_SIZE,
        config.IMG_SIZE
    )
    print(f"\n✅ Data generators created successfully!")
    print(f"   Training samples: {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Class mapping: {train_gen.class_indices}")

    # Step 4: Train the model
    print("\n" + "=" * 50)
    print("STEP 4: TRAINING MODEL")
    print("=" * 50)
    response = input("\nReady to start training? This may take a while. (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Training cancelled. You can run this script again later.")
        return

    model, history = train_model(
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        use_pretrained=True,
        fine_tune=True
    )

    print("\n" + "=" * 70)
    print(" 🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 70)
    print(f"\nModel saved at: {config.MODEL_PATH}")
    print("\nYou can now run the Flask API server:")
    print("   python api/app.py")
    print("\nOr test predictions:")
    print("   python src/predict.py")


if __name__ == "__main__":
    main()
