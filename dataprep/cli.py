import typer
from typing import Optional
from .utils import load_dataset, save_dataset
from .engine import DataPrepEngine
from .logger import DataLogger

app = typer.Typer(help="DataPrep AI: Autonomous Data Science Assistant")
log = DataLogger()

@app.command()
def auto(
    file: str = typer.Argument(..., help="Path to the dataset (CSV, Excel, JSON)"),
    goal: Optional[str] = typer.Option(None, "--goal", "-g", help="Goal: prediction, classification, analysis")
):
    """
    Automatically analyze, clean, and prepare the dataset.
    """
    try:
        # Load
        df = load_dataset(file)
        
        # Initialize Engine
        engine = DataPrepEngine(df)
        
        # Analyze
        engine.analyze()
        
        # Clean
        engine.clean()
        
        # Transform
        engine.transform(goal=goal)
        
        # Save
        result_df = engine.get_result()
        save_dataset(result_df, file)
        
        log.success("Data cleaned successfully")
        
    except Exception as e:
        # Error already logged in utils/engine if needed
        pass

@app.command()
def fix(
    instruction: str = typer.Argument(..., help="Natural language instruction"),
    file: str = typer.Argument(..., help="Path to the dataset")
):
    """
    Perform a specific fix using natural language (Experimental).
    """
    log.info(f"Processing command: '{instruction}' on {file}")
    # Simple heuristic-based mapping for demonstration
    if "missing" in instruction.lower():
        auto(file) # Reuse auto for now
    else:
        log.warn("Instruction not fully understood. Running default 'auto' mode.")
        auto(file)

if __name__ == "__main__":
    app()
