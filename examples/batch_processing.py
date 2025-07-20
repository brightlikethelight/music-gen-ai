#!/usr/bin/env python3
"""
Batch processing example for MusicGen Unified.

This example demonstrates how to process multiple music generation requests
efficiently using the batch processing capabilities.
"""

import pandas as pd
from pathlib import Path
from musicgen import BatchProcessor


def main():
    """Demonstrate batch processing capabilities."""
    # Create sample batch data
    batch_data = [
        {
            'prompt': 'upbeat electronic dance music',
            'duration': 15.0,
            'output_filename': 'edm_track.wav'
        },
        {
            'prompt': 'classical piano sonata',
            'duration': 20.0,
            'output_filename': 'classical_piano.wav'
        },
        {
            'prompt': 'jazz guitar improvisation',
            'duration': 18.0,
            'output_filename': 'jazz_guitar.wav'
        },
        {
            'prompt': 'ambient nature sounds with flute',
            'duration': 25.0,
            'output_filename': 'ambient_nature.wav'
        }
    ]
    
    # Create CSV file
    df = pd.DataFrame(batch_data)
    csv_path = "batch_example.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Created batch CSV: {csv_path}")
    print(f"Batch contains {len(batch_data)} tracks")
    
    # Initialize batch processor
    processor = BatchProcessor()
    
    # Set output directory
    output_dir = Path("batch_example_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing batch to: {output_dir}")
    
    # Process the batch
    try:
        results = processor.process_batch(
            csv_path=csv_path,
            output_dir=str(output_dir),
            max_workers=2,  # Adjust based on your system
            show_progress=True
        )
        
        print(f"\nBatch processing completed!")
        print(f"Successfully processed: {results['successful']} tracks")
        print(f"Failed: {results['failed']} tracks")
        
        # List generated files
        generated_files = list(output_dir.glob("*.wav"))
        print(f"\nGenerated files:")
        for file_path in generated_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"- {file_path.name}: {size_mb:.2f} MB")
            
    except Exception as e:
        print(f"Batch processing failed: {e}")
    
    # Cleanup
    print(f"\nCleanup: removing {csv_path}")
    Path(csv_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()