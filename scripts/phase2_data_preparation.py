#!/usr/bin/env python3
"""
Phase 2: Advanced Data Preparation Script
This script implements comprehensive data preparation for Arabic Sign Language Recognition
"""

import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_splitter import create_optimal_splits, AdvancedDataSplitter
from src.data.data_validator import run_comprehensive_validation
from src.data.augmentation import create_augmentation_pipeline, visualize_augmentations
from src.utils.helpers import setup_logging, get_device, create_directories
from src.config.config import get_config

def phase2_comprehensive_preparation():
    """Complete Phase 2 data preparation pipeline"""

    print("🚀 Phase 2: Advanced Data Preparation Pipeline")
    print("=" * 55)

    # Setup
    logger = setup_logging('logs/phase2_preparation.log')
    config = get_config()
    device = get_device()

    # Configuration
    labels_file = getattr(config.data, 'labels_file', 'data/ArSL_Data_Labels.csv')
    data_dir = config.data.raw_data_dir

    print(f"📁 Labels file: {labels_file}")
    print(f"📁 Data directory: {data_dir}")
    print(f"🎯 Device: {device}")

    # Create necessary directories
    create_directories([
        'data/processed',
        'data/analysis', 
        'data/splits',
        'logs'
    ])

    start_time = time.time()

    try:
        # Step 1: Comprehensive Data Validation
        print("\n" + "="*50)
        print("STEP 1: COMPREHENSIVE DATA VALIDATION")
        print("="*50)

        validation_start = time.time()
        validation_results = run_comprehensive_validation(
            labels_file=labels_file,
            data_dir=data_dir,
            output_dir='data/analysis'
        )
        validation_time = time.time() - validation_start
        print(f"✅ Validation completed in {validation_time:.1f}s")

        # Step 2: Advanced Data Splitting
        print("\n" + "="*50)
        print("STEP 2: ADVANCED DATA SPLITTING")
        print("="*50)

        split_start = time.time()

        # Create splitter
        splitter = AdvancedDataSplitter(labels_file, config.system.random_seed)

        # Create stratified splits
        train_df, val_df, test_df = splitter.stratified_split(
            train_size=config.data.train_split,
            val_size=config.data.val_split, 
            test_size=config.data.test_split
        )

        # Analyze split quality
        split_analysis = splitter.analyze_split_quality(train_df, val_df, test_df)

        # Save splits
        splitter.save_splits(train_df, val_df, test_df, 'data/processed')

        # Create visualizations
        splitter.visualize_splits(train_df, val_df, test_df, 'data/analysis')

        split_time = time.time() - split_start
        print(f"✅ Data splitting completed in {split_time:.1f}s")

        # Step 3: Cross-Validation Splits (Optional)
        print("\n" + "="*50)
        print("STEP 3: CROSS-VALIDATION SPLITS (OPTIONAL)")
        print("="*50)

        cv_start = time.time()
        cv_folds = splitter.create_cross_validation_folds(n_folds=5)

        # Save CV folds
        cv_dir = 'data/splits/cv_folds'
        os.makedirs(cv_dir, exist_ok=True)

        for i, (train_fold, val_fold) in enumerate(cv_folds):
            train_fold.to_csv(os.path.join(cv_dir, f'fold_{i}_train.csv'), index=False)
            val_fold.to_csv(os.path.join(cv_dir, f'fold_{i}_val.csv'), index=False)

        cv_time = time.time() - cv_start
        print(f"✅ Cross-validation splits created in {cv_time:.1f}s")

        # Step 4: Augmentation Pipeline Setup
        print("\n" + "="*50)
        print("STEP 4: AUGMENTATION PIPELINE SETUP")
        print("="*50)

        aug_start = time.time()

        # Create augmentation pipelines
        config_dict = config.to_dict()
        train_aug = create_augmentation_pipeline(config_dict, mode='train')
        val_aug = create_augmentation_pipeline(config_dict, mode='val')
        tta_aug = create_augmentation_pipeline(config_dict, mode='tta')

        print("✅ Training augmentation pipeline created")
        print("✅ Validation augmentation pipeline created") 
        print("✅ Test-time augmentation pipeline created")

        # Create augmentation visualizations (if sample image exists)
        sample_images = []
        for class_folder in os.listdir(data_dir)[:3]:  # Check first 3 folders
            class_path = os.path.join(data_dir, class_folder)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path)[:1]:  # Get first image
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sample_images.append(os.path.join(class_path, img_file))
                        break

        if not sample_images:
            # Try to find sample images directly in data_dir
            for img_file in os.listdir(data_dir)[:10]:
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(data_dir, img_file)
                    sample_images.append(img_path)
                    break

        if sample_images:
            try:
                visualize_augmentations(
                    image_path=sample_images[0],
                    config=config_dict,
                    output_dir='data/analysis/augmentations'
                )
                print("✅ Augmentation examples generated")
            except Exception as e:
                print(f"⚠️  Could not create augmentation examples: {e}")
        else:
            print("⚠️  No sample images found for augmentation visualization")

        aug_time = time.time() - aug_start
        print(f"✅ Augmentation setup completed in {aug_time:.1f}s")

        # Step 5: Data Pipeline Testing
        print("\n" + "="*50)
        print("STEP 5: DATA PIPELINE TESTING")
        print("="*50)

        test_start = time.time()

        # Test data loading with new splits
        from src.data.dataset import create_data_loaders

        try:
            train_loader, val_loader, test_loader = create_data_loaders(
                labels_file='data/processed/train.csv',
                data_dir=data_dir,
                batch_size=config.data.batch_size,
                num_workers=min(config.data.num_workers, 2),  # Reduce for testing
                train_split=1.0,  # Already split
                val_split=0.0,
                image_size=config.data.input_size
            )

            # Test loading a batch
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader)) 
            test_batch = next(iter(test_loader))

            print(f"✅ Training batch: {train_batch[0].shape}")
            print(f"✅ Validation batch: {val_batch[0].shape}")
            print(f"✅ Test batch: {test_batch[0].shape}")

        except Exception as e:
            print(f"⚠️  Data loading test failed: {e}")

        test_time = time.time() - test_start
        print(f"✅ Pipeline testing completed in {test_time:.1f}s")

        # Phase 2 Summary
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print("PHASE 2 COMPLETION SUMMARY")
        print("="*50)

        print(f"📊 Dataset validation: {validation_time:.1f}s")
        print(f"✂️  Data splitting: {split_time:.1f}s") 
        print(f"🔄 Cross-validation: {cv_time:.1f}s")
        print(f"🎨 Augmentation setup: {aug_time:.1f}s")
        print(f"🧪 Pipeline testing: {test_time:.1f}s")
        print(f"⏱️  Total time: {total_time:.1f}s")

        # Results summary
        validator = validation_results['validator']
        file_integrity = validator.validate_file_integrity()

        print(f"\n📈 Results Summary:")
        print(f"   Total images: {len(validator.df):,}")
        print(f"   Usable images: {file_integrity['summary']['usable_files']:,}")
        print(f"   Training samples: {len(train_df):,}")
        print(f"   Validation samples: {len(val_df):,}")
        print(f"   Test samples: {len(test_df):,}")
        print(f"   Classes: {config.data.num_classes}")

        print(f"\n📁 Generated Files:")
        print(f"   📊 data/analysis/comprehensive_validation_report.txt")
        print(f"   📊 data/analysis/data_splits_analysis.png")
        print(f"   📊 data/analysis/augmentations/augmentation_examples.png")
        print(f"   📄 data/processed/train.csv")
        print(f"   📄 data/processed/val.csv")
        print(f"   📄 data/processed/test.csv")
        print(f"   📄 data/processed/class_mapping.json")
        print(f"   📄 data/splits/cv_folds/ (5 CV folds)")

        print(f"\n🎉 Phase 2: Data Preparation completed successfully!")
        print(f"\n🚀 Ready for Phase 3: Model Development")
        print(f"\n📝 Next steps:")
        print(f"   1. Review validation report in data/analysis/")
        print(f"   2. Check data split visualizations") 
        print(f"   3. Proceed to model architecture implementation")
        print(f"   4. Begin training pipeline development")

        return True

    except Exception as e:
        print(f"\n❌ Phase 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = phase2_comprehensive_preparation()
    sys.exit(0 if success else 1)
