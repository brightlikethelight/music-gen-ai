"""
Unit tests for musicgen.services.batch module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from musicgen.services.batch import BatchProcessor, create_sample_csv


class TestBatchProcessor:
    """Test BatchProcessor class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_csv_file(self, temp_dir):
        """Create sample CSV file for testing."""
        csv_path = temp_dir / "test.csv"
        data = [
            {"prompt": "piano music", "duration": 30, "output_file": "output1.mp3"},
            {"prompt": "guitar solo", "duration": 45, "output_file": "output2.mp3"},
            {"prompt": "drum beat", "duration": 20},  # No output_file
        ]
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_init_default_params(self, temp_dir):
        """Test initialization with default parameters."""
        processor = BatchProcessor(output_dir=str(temp_dir))
        
        assert processor.output_dir == temp_dir
        assert processor.max_workers >= 1
        assert processor.model_name == "facebook/musicgen-small"
        assert processor.device is None
        assert temp_dir.exists()

    def test_init_custom_params(self, temp_dir):
        """Test initialization with custom parameters."""
        output_dir = temp_dir / "custom_output"
        processor = BatchProcessor(
            output_dir=str(output_dir),
            max_workers=2,
            model_name="facebook/musicgen-medium",
            device="cpu"
        )
        
        assert processor.output_dir == output_dir
        assert processor.max_workers == 2
        assert processor.model_name == "facebook/musicgen-medium"
        assert processor.device == "cpu"
        assert output_dir.exists()

    @patch("os.cpu_count")
    def test_init_auto_workers(self, mock_cpu_count, temp_dir):
        """Test automatic worker count detection."""
        mock_cpu_count.return_value = 8
        processor = BatchProcessor(output_dir=str(temp_dir))
        assert processor.max_workers == 4  # min(8, 4)

        mock_cpu_count.return_value = 2
        processor = BatchProcessor(output_dir=str(temp_dir))
        assert processor.max_workers == 2  # min(2, 4)

    def test_load_csv_valid(self, sample_csv_file, temp_dir):
        """Test loading valid CSV file."""
        processor = BatchProcessor(output_dir=str(temp_dir))
        jobs = processor.load_csv(sample_csv_file)
        
        assert len(jobs) == 3
        
        # Check first job
        job1 = jobs[0]
        assert job1["id"] == 0
        assert job1["prompt"] == "piano music"
        assert job1["duration"] == 30.0
        assert job1["output_file"].endswith("output1.mp3")
        assert job1["temperature"] == 1.0
        assert job1["guidance_scale"] == 3.0

        # Check job with auto-generated output file  
        job3 = jobs[2]
        assert "output_" in job3["output_file"] and job3["output_file"].endswith(".mp3")

    def test_load_csv_missing_file(self, temp_dir):
        """Test loading non-existent CSV file."""
        processor = BatchProcessor(output_dir=str(temp_dir))
        
        with pytest.raises(FileNotFoundError):
            processor.load_csv("nonexistent.csv")

    def test_load_csv_missing_prompt_column(self, temp_dir):
        """Test loading CSV without required prompt column."""
        csv_path = temp_dir / "invalid.csv"
        data = [{"duration": 30, "output_file": "output.mp3"}]
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        processor = BatchProcessor(output_dir=str(temp_dir))
        
        with pytest.raises(ValueError, match="CSV must have 'prompt' column"):
            processor.load_csv(str(csv_path))

    def test_load_csv_invalid_data(self, temp_dir):
        """Test loading CSV with invalid data."""
        csv_path = temp_dir / "invalid_data.csv"
        data = [
            {"prompt": "", "duration": 30},  # Empty prompt
            {"prompt": "valid music", "duration": 200},  # Invalid duration
            {"prompt": "another valid", "duration": -5},  # Negative duration
        ]
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        processor = BatchProcessor(output_dir=str(temp_dir))
        jobs = processor.load_csv(str(csv_path))
        
        # Should get jobs with duration corrections (empty prompt may become "nan")
        assert len(jobs) >= 2
        
        # Find the valid jobs (excluding any that became "nan")
        valid_jobs = [job for job in jobs if job["prompt"] not in ["", "nan"]]
        assert len(valid_jobs) == 2
        
        # Check the valid jobs have corrected durations
        valid_music_job = next(job for job in valid_jobs if "valid music" in job["prompt"])
        another_valid_job = next(job for job in valid_jobs if "another valid" in job["prompt"])
        
        assert valid_music_job["duration"] == 10.0  # Corrected from 200
        assert another_valid_job["duration"] == 10.0  # Corrected from -5

    @patch("musicgen.services.batch.MusicGenerator")
    def test_process_single_success(self, mock_generator_class, temp_dir):
        """Test successful single job processing."""
        # Mock generator
        mock_generator = MagicMock()
        mock_generator.generate.return_value = ("fake_audio", 32000)
        mock_generator.save_audio.return_value = "/path/to/output.mp3"
        mock_generator_class.return_value = mock_generator
        
        # Mock file size
        with patch("os.path.getsize", return_value=1024000):
            processor = BatchProcessor(output_dir=str(temp_dir))
            job = {
                "id": 1,
                "prompt": "test music",
                "output_file": "test.mp3",
                "duration": 30.0,
                "temperature": 1.0,
                "guidance_scale": 3.0,
            }
            
            result = processor.process_single(job)
            
            assert result["success"] is True
            assert result["id"] == 1
            assert result["prompt"] == "test music"
            assert result["output_file"] == "/path/to/output.mp3"
            assert result["error"] is None
            assert result["generation_time"] > 0
            assert result["file_size"] == 1024000
            
            # Verify generator was called correctly
            mock_generator.generate.assert_called_once_with(
                prompt="test music",
                duration=30.0,
                temperature=1.0,
                guidance_scale=3.0,
            )

    @patch("musicgen.services.batch.MusicGenerator")
    def test_process_single_failure(self, mock_generator_class, temp_dir):
        """Test failed single job processing."""
        # Mock generator to raise exception
        mock_generator = MagicMock()
        mock_generator.generate.side_effect = Exception("Generation failed")
        mock_generator_class.return_value = mock_generator
        
        processor = BatchProcessor(output_dir=str(temp_dir))
        job = {
            "id": 1,
            "prompt": "test music",
            "output_file": "test.mp3",
            "duration": 30.0,
            "temperature": 1.0,
            "guidance_scale": 3.0,
        }
        
        result = processor.process_single(job)
        
        assert result["success"] is False
        assert result["error"] == "Generation failed"
        assert result["generation_time"] == 0

    def test_process_batch_empty(self, temp_dir):
        """Test processing empty batch."""
        processor = BatchProcessor(output_dir=str(temp_dir))
        results = processor.process_batch([])
        assert results == []

    @patch("musicgen.services.batch.ProcessPoolExecutor")
    def test_process_batch_with_jobs(self, mock_executor_class, temp_dir):
        """Test processing batch with jobs."""
        # Mock executor
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Mock futures
        mock_future1 = MagicMock()
        mock_future1.result.return_value = {"id": 1, "success": True}
        mock_future2 = MagicMock()
        mock_future2.result.return_value = {"id": 2, "success": True}
        
        mock_executor.submit.side_effect = [mock_future1, mock_future2]
        
        # Mock as_completed
        with patch("musicgen.services.batch.as_completed", return_value=[mock_future1, mock_future2]):
            processor = BatchProcessor(output_dir=str(temp_dir), max_workers=2)
            jobs = [{"id": 1}, {"id": 2}]
            
            results = processor.process_batch(jobs)
            
            assert len(results) == 2
            assert results[0]["id"] == 1
            assert results[1]["id"] == 2

    def test_process_batch_with_progress_callback(self, temp_dir):
        """Test batch processing with progress callback."""
        processor = BatchProcessor(output_dir=str(temp_dir))
        callback_calls = []
        
        def progress_callback(current, total, message):
            callback_calls.append((current, total, message))
        
        # Mock the ProcessPoolExecutor to avoid actual processing
        with patch("musicgen.services.batch.ProcessPoolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            
            mock_future = MagicMock()
            mock_future.result.return_value = {"id": 1, "success": True}
            mock_executor.submit.return_value = mock_future
            
            with patch("musicgen.services.batch.as_completed", return_value=[mock_future]):
                jobs = [{"id": 1}]
                processor.process_batch(jobs, progress_callback)
                
                assert len(callback_calls) == 1
                assert callback_calls[0] == (1, 1, "Processing 1/1")

    def test_save_results(self, temp_dir):
        """Test saving results to JSON."""
        processor = BatchProcessor(output_dir=str(temp_dir))
        results = [
            {"id": 1, "success": True, "generation_time": 5.0},
            {"id": 2, "success": False, "generation_time": 0},
            {"id": 3, "success": True, "generation_time": 3.0},
        ]
        
        summary = processor.save_results(results, "test_results.json")
        
        # Check summary
        assert summary["total_jobs"] == 3
        assert summary["successful"] == 2
        assert summary["failed"] == 1
        assert summary["success_rate"] == 2/3
        assert summary["total_generation_time"] == 8.0
        assert "timestamp" in summary
        
        # Check file was created
        results_file = temp_dir / "test_results.json"
        assert results_file.exists()
        
        # Check file contents
        with open(results_file, "r") as f:
            data = json.load(f)
        
        assert "summary" in data
        assert "results" in data
        assert data["summary"]["total_jobs"] == 3
        assert len(data["results"]) == 3

    def test_save_results_empty(self, temp_dir):
        """Test saving empty results."""
        processor = BatchProcessor(output_dir=str(temp_dir))
        summary = processor.save_results([])
        
        assert summary["total_jobs"] == 0
        assert summary["success_rate"] == 0


class TestCreateSampleCSV:
    """Test create_sample_csv function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_create_sample_csv_default(self, temp_dir):
        """Test creating sample CSV with default filename."""
        with patch("musicgen.services.batch.logger"):
            # Change to temp directory to control where file is created
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                filename = create_sample_csv()
                
                assert filename == "sample_batch.csv"
                assert (temp_dir / filename).exists()
                
                # Check contents
                df = pd.read_csv(temp_dir / filename)
                assert len(df) == 3
                assert "prompt" in df.columns
                assert "duration" in df.columns
                assert "output_file" in df.columns
                
            finally:
                os.chdir(original_cwd)

    def test_create_sample_csv_custom_filename(self, temp_dir):
        """Test creating sample CSV with custom filename."""
        custom_filename = str(temp_dir / "custom_sample.csv")
        
        with patch("musicgen.services.batch.logger"):
            filename = create_sample_csv(custom_filename)
            
            assert filename == custom_filename
            assert Path(custom_filename).exists()
            
            # Check contents
            df = pd.read_csv(custom_filename)
            assert len(df) == 3
            assert df.iloc[0]["prompt"] == "upbeat jazz piano"
            assert df.iloc[1]["prompt"] == "ambient electronic soundscape"
            assert df.iloc[2]["prompt"] == "classical string quartet"

    def test_create_sample_csv_content_validation(self, temp_dir):
        """Test that sample CSV has valid content."""
        csv_file = str(temp_dir / "validation_test.csv")
        
        with patch("musicgen.services.batch.logger"):
            create_sample_csv(csv_file)
            
            df = pd.read_csv(csv_file)
            
            # Validate each row
            for _, row in df.iterrows():
                assert isinstance(row["prompt"], str)
                assert len(row["prompt"]) > 0
                assert isinstance(row["duration"], (int, float))
                assert row["duration"] > 0
                assert isinstance(row["output_file"], str)
                assert row["output_file"].endswith(".mp3")