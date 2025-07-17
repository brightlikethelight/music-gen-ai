"""
Command-line interface for MusicGen Unified.
Simple, clean, focused on what matters.
"""

import os
import sys
import time
import logging
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.logging import RichHandler

from .generator import MusicGenerator
from .batch import BatchProcessor, create_sample_csv
from .prompt import PromptEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

app = typer.Typer(
    name="musicgen",
    help="MusicGen Unified - Simple instrumental music generation"
)
console = Console()


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Music description"),
    output: str = typer.Option("output.mp3", "-o", "--output", help="Output file"),
    duration: float = typer.Option(30.0, "-d", "--duration", help="Duration in seconds"),
    model: str = typer.Option("small", "-m", "--model", help="Model size (small/medium/large)"),
    temperature: float = typer.Option(1.0, "-t", "--temperature", help="Sampling temperature"),
    guidance: float = typer.Option(3.0, "-g", "--guidance", help="Guidance scale"),
    device: Optional[str] = typer.Option(None, "--device", help="Device (cuda/cpu)"),
    no_optimize: bool = typer.Option(False, "--no-optimize", help="Disable GPU optimization")
):
    """Generate music from text description."""
    
    # Validate duration
    if duration <= 0 or duration > 300:
        rprint("[red]Error: Duration must be between 0 and 300 seconds[/red]")
        raise typer.Exit(1)
    
    # Model mapping
    model_map = {
        "small": "facebook/musicgen-small",
        "medium": "facebook/musicgen-medium",
        "large": "facebook/musicgen-large"
    }
    
    if model not in model_map:
        rprint(f"[red]Error: Model must be one of: {', '.join(model_map.keys())}[/red]")
        raise typer.Exit(1)
    
    try:
        # Initialize generator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            generator = MusicGenerator(
                model_map[model],
                device=device,
                optimize=not no_optimize
            )
        
        # Show info
        info = generator.get_info()
        rprint(f"[green]✓ Model loaded[/green]")
        rprint(f"  Device: {info['device']}")
        if 'gpu' in info:
            rprint(f"  GPU: {info['gpu']}")
        
        # Improve prompt
        engineer = PromptEngineer()
        is_valid, issues = engineer.validate_prompt(prompt)
        
        if issues:
            rprint("\n[yellow]Prompt issues:[/yellow]")
            for issue in issues:
                rprint(f"  ⚠️  {issue}")
        
        improved = engineer.improve_prompt(prompt)
        if improved != prompt:
            rprint(f"\n[cyan]Improved prompt: {improved}[/cyan]")
            if typer.confirm("Use improved prompt?", default=True):
                prompt = improved
        
        # Generate with progress
        rprint(f"\n[yellow]Generating {duration}s of music...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Generating...", total=100)
            
            def progress_callback(percent, message):
                progress.update(task, completed=percent, description=message)
            
            start_time = time.time()
            audio, sample_rate = generator.generate(
                prompt,
                duration,
                temperature,
                guidance,
                progress_callback
            )
        
        # Save audio
        output_path = generator.save_audio(audio, sample_rate, output)
        gen_time = time.time() - start_time
        
        # Show results
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        rprint(f"\n[green]✅ Success![/green]")
        rprint(f"  Output: {output_path} ({file_size:.1f} MB)")
        rprint(f"  Duration: {duration}s")
        rprint(f"  Time: {gen_time:.1f}s ({duration/gen_time:.1f}x realtime)")
        
    except KeyboardInterrupt:
        rprint("\n[yellow]Generation cancelled[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def batch(
    csv_file: str = typer.Argument(..., help="CSV file with prompts"),
    output_dir: str = typer.Option("batch_output", "-o", "--output-dir", help="Output directory"),
    workers: int = typer.Option(None, "-w", "--workers", help="Number of workers"),
    model: str = typer.Option("small", "-m", "--model", help="Model size"),
    device: Optional[str] = typer.Option(None, "--device", help="Device (cuda/cpu)"),
    results: str = typer.Option("results.json", "-r", "--results", help="Results file")
):
    """Process multiple generations from CSV file."""
    
    model_map = {
        "small": "facebook/musicgen-small",
        "medium": "facebook/musicgen-medium",
        "large": "facebook/musicgen-large"
    }
    
    try:
        # Initialize processor
        processor = BatchProcessor(
            output_dir=output_dir,
            max_workers=workers,
            model_name=model_map.get(model, model_map["small"]),
            device=device
        )
        
        # Load jobs
        rprint(f"[cyan]Loading jobs from {csv_file}...[/cyan]")
        jobs = processor.load_csv(csv_file)
        
        if not jobs:
            rprint("[red]No valid jobs found[/red]")
            raise typer.Exit(1)
        
        rprint(f"[green]✓ Loaded {len(jobs)} jobs[/green]")
        
        # Process with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Processing...", total=len(jobs))
            
            def progress_callback(current, total, message):
                progress.update(task, completed=current, description=message)
            
            start_time = time.time()
            results_list = processor.process_batch(jobs, progress_callback)
        
        # Save results
        summary = processor.save_results(results_list, results)
        
        # Show summary
        table = Table(title="Batch Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Jobs", str(summary['total_jobs']))
        table.add_row("Successful", str(summary['successful']))
        table.add_row("Failed", str(summary['failed']))
        table.add_row("Success Rate", f"{summary['success_rate']*100:.1f}%")
        table.add_row("Total Time", f"{time.time() - start_time:.1f}s")
        
        console.print(table)
        
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def prompt(
    text: Optional[str] = typer.Argument(None, help="Prompt to improve"),
    examples: bool = typer.Option(False, "-e", "--examples", help="Show examples"),
    validate: bool = typer.Option(False, "-v", "--validate", help="Validate prompt"),
    variations: int = typer.Option(0, "--variations", help="Generate variations")
):
    """Improve prompts for better results."""
    
    engineer = PromptEngineer()
    
    if examples:
        rprint("[cyan]Example prompts:[/cyan]")
        for example in engineer.get_examples():
            rprint(f"  • {example}")
        return
    
    if not text:
        text = typer.prompt("Enter prompt to improve")
    
    # Validate
    if validate:
        is_valid, issues = engineer.validate_prompt(text)
        if is_valid:
            rprint("[green]✓ Prompt is valid[/green]")
        else:
            rprint("[red]Issues found:[/red]")
            for issue in issues:
                rprint(f"  ⚠️  {issue}")
    
    # Improve
    improved = engineer.improve_prompt(text)
    if improved != text:
        rprint(f"\n[green]Improved:[/green] {improved}")
    else:
        rprint("[green]✓ Prompt is already good[/green]")
    
    # Variations
    if variations > 0:
        rprint(f"\n[cyan]Variations:[/cyan]")
        for var in engineer.suggest_variations(improved, variations):
            rprint(f"  • {var}")


@app.command()
def serve(
    port: int = typer.Option(8080, "-p", "--port", help="Port to serve on"),
    host: str = typer.Option("0.0.0.0", "-h", "--host", help="Host to bind to")
):
    """Start web interface."""
    
    rprint(f"[yellow]Starting web server on http://{host}:{port}[/yellow]")
    
    # Import here to avoid circular imports
    from .web import create_app
    
    app = create_app()
    
    try:
        import uvicorn
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        rprint("[red]Error: Web server requires uvicorn[/red]")
        rprint("Install with: pip install musicgen-unified[web]")
        raise typer.Exit(1)


@app.command()
def api(
    port: int = typer.Option(8000, "-p", "--port", help="Port to serve on"),
    host: str = typer.Option("0.0.0.0", "-h", "--host", help="Host to bind to"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of workers")
):
    """Start REST API server."""
    
    rprint(f"[yellow]Starting API server on http://{host}:{port}[/yellow]")
    rprint(f"[dim]API docs: http://{host}:{port}/docs[/dim]")
    
    # Import here to avoid circular imports
    from .api import app
    
    try:
        import uvicorn
        uvicorn.run(
            "musicgen.api:app",
            host=host,
            port=port,
            workers=workers,
            reload=False
        )
    except ImportError:
        rprint("[red]Error: API server requires uvicorn[/red]")
        rprint("Install with: pip install musicgen-unified[api]")
        raise typer.Exit(1)


@app.command()
def info():
    """Show system information."""
    
    table = Table(title="MusicGen Unified - System Info")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    
    # Python version
    table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}")
    
    # PyTorch
    try:
        import torch
        table.add_row("PyTorch", torch.__version__)
        table.add_row("CUDA Available", "✓" if torch.cuda.is_available() else "✗")
        
        if torch.cuda.is_available():
            table.add_row("GPU", torch.cuda.get_device_name())
            table.add_row("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        table.add_row("PyTorch", "Not installed")
    
    # Models
    table.add_row("Models", "")
    table.add_row("  Small", "300M params (fastest)")
    table.add_row("  Medium", "1.5B params (balanced)")
    table.add_row("  Large", "3.3B params (best quality)")
    
    console.print(table)


@app.command("create-sample-csv")
def create_sample():
    """Create sample CSV for batch processing."""
    
    filename = "sample_batch.csv"
    create_sample_csv(filename)
    
    rprint(f"[green]✓ Created {filename}[/green]")
    rprint(f"\nTo process: [cyan]musicgen batch {filename}[/cyan]")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()