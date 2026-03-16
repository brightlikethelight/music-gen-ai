"""
Batch processing for music generation.
Simple, efficient, no over-engineering.
"""

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..core.generator import MusicGenerator

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Process multiple music generation requests from CSV.

    CSV Format:
    - prompt: Text description (required)
    - duration: Duration in seconds (optional, default: 10)
    - output_file: Output filename (optional, auto-generated)
    - temperature: Sampling temperature (optional)
    - guidance_scale: Guidance scale (optional)
    """

    def __init__(
        self,
        output_dir: str = "batch_output",
        max_workers: Optional[int] = None,
        model_name: str = "facebook/musicgen-small",
        device: Optional[str] = None,
    ):
        """
        Initialize batch processor.

        Args:
            output_dir: Directory for output files
            max_workers: Number of parallel workers (auto-detect if None)
            model_name: Model to use
            device: Device (auto-detect if None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers or min(os.cpu_count() or 1, 4)
        self.model_name = model_name
        self.device = device

        logger.info(
            "Batch processor ready: %s workers, output: %s", self.max_workers, self.output_dir
        )

    def load_csv(self, csv_file: str) -> List[Dict[str, Any]]:
        """Load and validate CSV file."""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        # Load with pandas for better handling
        df = pd.read_csv(csv_file)

        # Validate required columns
        if "prompt" not in df.columns:
            raise ValueError("CSV must have 'prompt' column")

        # Convert to list of jobs
        jobs = []
        for idx, row in df.iterrows():
            job = {
                "id": idx,
                "prompt": str(row["prompt"]).strip(),
                "duration": float(row.get("duration", 10.0)),
                "output_file": (
                    str(row["output_file"])
                    if "output_file" in row and pd.notna(row["output_file"])
                    else f"output_{idx:04d}.mp3"
                ),
                "temperature": float(row.get("temperature", 1.0)),
                "guidance_scale": float(row.get("guidance_scale", 3.0)),
            }

            # Validate
            if not job["prompt"]:
                logger.warning("Row %s: Empty prompt, skipping", idx)
                continue

            if not (0.1 <= job["duration"] <= 120):
                logger.warning("Row %s: Invalid duration, using 10s", idx)
                job["duration"] = 10.0

            # Ensure output path
            if not os.path.isabs(job["output_file"]):
                job["output_file"] = str(self.output_dir / job["output_file"])

            jobs.append(job)

        logger.info("Loaded %s valid jobs from %s", len(jobs), csv_file)
        return jobs

    def process_single(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single generation job."""
        start_time = time.time()
        result = {
            "id": job["id"],
            "prompt": job["prompt"],
            "output_file": job["output_file"],
            "success": False,
            "error": None,
            "duration": job["duration"],
            "generation_time": 0,
        }

        try:
            # Create generator (each process gets its own)
            generator = MusicGenerator(self.model_name, self.device)

            # Generate audio
            audio, sample_rate = generator.generate(
                prompt=job["prompt"],
                duration=job["duration"],
                temperature=job["temperature"],
                guidance_scale=job["guidance_scale"],
            )

            # Save audio
            output_path = generator.save_audio(audio, sample_rate, job["output_file"])

            # Update result
            result["success"] = True
            result["output_file"] = output_path
            result["generation_time"] = time.time() - start_time
            result["file_size"] = os.path.getsize(output_path)

            logger.info("✓ Job %s: Generated in %.1fs", job["id"], result["generation_time"])

        except Exception as e:
            result["error"] = str(e)
            logger.error("✗ Job %s failed: %s", job["id"], e)

        return result

    def process_batch(
        self,
        jobs: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process batch of jobs.

        Args:
            jobs: List of job dictionaries
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of results
        """
        if not jobs:
            return []

        results = []
        total = len(jobs)

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {executor.submit(self.process_single, job): job for job in jobs}

            # Process completions
            for i, future in enumerate(as_completed(future_to_job)):
                if progress_callback:
                    progress_callback(i + 1, total, f"Processing {i+1}/{total}")

                try:
                    result = future.result(timeout=300)  # 5 min timeout
                    results.append(result)
                except Exception as e:
                    job = future_to_job[future]
                    results.append(
                        {
                            "id": job["id"],
                            "prompt": job["prompt"],
                            "output_file": job["output_file"],
                            "success": False,
                            "error": f"Processing failed: {e}",
                            "duration": job["duration"],
                            "generation_time": 0,
                        }
                    )

        return results

    def save_results(
        self, results: List[Dict[str, Any]], filename: str = "results.json"
    ) -> Dict[str, Any]:
        """Save batch results to JSON."""
        output_path = self.output_dir / filename

        # Calculate summary
        successful = sum(1 for r in results if r["success"])
        total_time = sum(r["generation_time"] for r in results if r["success"])

        summary = {
            "total_jobs": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "success_rate": successful / len(results) if results else 0,
            "total_generation_time": total_time,
            "timestamp": time.time(),
        }

        output_data = {"summary": summary, "results": results}

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info("Results saved to %s", output_path)
        return summary


def create_sample_csv(filename: str = "sample_batch.csv") -> str:
    """Create a sample CSV file for batch processing."""
    sample_data = [
        {"prompt": "upbeat jazz piano", "duration": 30, "output_file": "jazz_piano.mp3"},
        {"prompt": "ambient electronic soundscape", "duration": 45, "output_file": "ambient.mp3"},
        {"prompt": "classical string quartet", "duration": 60, "output_file": "classical.mp3"},
    ]

    df = pd.DataFrame(sample_data)
    df.to_csv(filename, index=False)

    logger.info("Sample CSV created: %s", filename)
    return filename
