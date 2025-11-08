"""
Helper script to download and prepare dataset from Kaggle.
This is an example - you'll need to replace with actual dataset.
"""
import os
import subprocess
import shutil


def download_kaggle_dataset():
    """
    Download dataset from Kaggle.
    
    Prerequisites:
  1. Install kaggle: pip install kaggle
    2. Setup Kaggle API credentials: https://www.kaggle.com/docs/api
    3. Place kaggle.json in ~/.kaggle/ or C:\Users\<user>\.kaggle\
 """
    
    print("=" * 60)
    print("  KAGGLE DATASET DOWNLOADER")
    print("=" * 60)
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("\n? Kaggle API is installed")
    except ImportError:
        print("\n? Kaggle API not installed!")
        print("\nTo install: pip install kaggle")
        print("Then setup credentials: https://www.kaggle.com/docs/api")
  return False
    
    # Example dataset - replace with actual robot vs human dataset
    # Search on Kaggle for: "robot vs human", "humanoid detection", etc.
    
    print("\n?? Suggested Datasets on Kaggle:")
    print("1. Search for: 'robot vs human classification'")
    print("2. Or use a combination of:")
 print("   - Human detection datasets (COCO, CelebA, etc.)")
    print("   - Robot/android images datasets")
    
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 60)
    
    print("\nOption 1: Download from Kaggle manually")
    print("1. Go to: https://www.kaggle.com/datasets")
    print("2. Search for 'robot human classification' or similar")
    print("3. Download the dataset")
    print("4. Extract to: data/raw/")
    print("5. Organize into:")
    print("   - data/raw/human/")
    print("   - data/raw/robot/")
    
    print("\nOption 2: Use combination of datasets")
    print("Humans:")
    print("  - CelebA: https://www.kaggle.com/jessicali9530/celeba-dataset")
    print("  - COCO Person: https://cocodataset.org/")
    print("Robots:")
    print("  - Search 'robot images', 'android', 'humanoid robot'")
    print("  - Or scrape from Google Images (with proper licensing)")
    
    print("\n" + "=" * 60)
    
    # Example code for automatic download (uncomment and modify)
    """
    dataset_name = "username/dataset-name"  # Replace with actual dataset
    
    try:
        print(f"\nDownloading dataset: {dataset_name}")
      kaggle.api.dataset_download_files(
            dataset_name,
         path='data/raw',
            unzip=True
     )
     print("? Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"? Error downloading dataset: {e}")
     return False
    """
 
    return False


def organize_dataset():
    """
    Organize downloaded dataset into proper structure.
    """
    print("\n" + "=" * 60)
    print("  ORGANIZING DATASET")
    print("=" * 60)
    
    raw_dir = "data/raw"
    human_dir = os.path.join(raw_dir, "human")
    robot_dir = os.path.join(raw_dir, "robot")
    
    # Create directories
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(robot_dir, exist_ok=True)
    
    print(f"\n? Created directories:")
    print(f"  - {human_dir}")
    print(f"  - {robot_dir}")
    
    print("\n?? Instructions:")
  print("1. Place all human images in: data/raw/human/")
    print("2. Place all robot images in: data/raw/robot/")
    print("3. Supported formats: JPG, PNG, TIFF, RAW (ARW, CR2, NEF, etc.)")
    print("4. After organizing, run: python src/data_preprocessing.py")
    
    return True


def create_sample_dataset():
    """
    Create a minimal sample dataset for testing (using test images).
    """
    print("\n" + "=" * 60)
    print("  CREATING SAMPLE DATASET FOR TESTING")
    print("=" * 60)
    
    images_dir = "Images"
    raw_dir = "data/raw"
    
    if not os.path.exists(images_dir):
 print(f"\n? {images_dir} directory not found!")
        return False
 
    # Get test images
    test_images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    
    if not test_images:
        print(f"\n? No images found in {images_dir}")
        return False
    
    print(f"\nFound {len(test_images)} test images")
    
    # Organize by name (assuming naming convention: om_X, robot_X)
    for img in test_images:
        src = os.path.join(images_dir, img)
        
  if 'om' in img.lower() or 'human' in img.lower():
         dst_dir = os.path.join(raw_dir, "human")
        elif 'robot' in img.lower():
      dst_dir = os.path.join(raw_dir, "robot")
  else:
   continue
        
        os.makedirs(dst_dir, exist_ok=True)
  dst = os.path.join(dst_dir, img)
        
  try:
   shutil.copy2(src, dst)
            print(f"  Copied: {img} ? {dst_dir}")
        except Exception as e:
          print(f"  Error copying {img}: {e}")
    
    print("\n? Sample dataset created!")
    print("\n??  Note: This is a very small dataset for testing only.")
    print("For actual training, you need a larger dataset (100+ images per class)")
    
    return True


if __name__ == "__main__":
    print("""
    ????????????????????????????????????????????????????????????
    ?             ?
    ?            ?? Dataset Preparation Helper        ?
    ?       ?
  ????????????????????????????????????????????????????????????
    """)
    
    print("\nWhat would you like to do?")
    print("1. Download from Kaggle (requires setup)")
    print("2. Organize existing dataset")
    print("3. Create sample dataset from test images (for testing only)")
    print("0. Exit")
    
    choice = input("\nEnter your choice: ").strip()
    
    if choice == "1":
        download_kaggle_dataset()
    elif choice == "2":
        organize_dataset()
    elif choice == "3":
        create_sample_dataset()
    elif choice == "0":
        print("\n?? Goodbye!")
    else:
  print("\n? Invalid choice!")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
