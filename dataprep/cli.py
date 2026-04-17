import typer
from typing import Optional
from .utils import load_dataset, save_dataset
from .engine import DataPrepEngine
from .logger import DataLogger
from .dl import Sequential, Dense, ReLU, Sigmoid, Tanh, Softmax
from .dl.layers import ActivationLayer

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

# DL Commands
dl_app = typer.Typer(help="Deep Learning Tools: Build and test neural networks")
app.add_typer(dl_app, name="dl")

@dl_app.command("build")
def build_model():
    """
    Interactively build a neural network.
    """
    log.info("--- [Deep Learning Model Builder] ---")
    
    try:
        num_layers = int(typer.prompt("How many dense layers?"))
        input_dim = int(typer.prompt("Input feature dimension (e.g., number of columns)?"))
        
        model = Sequential()
        current_dim = input_dim
        
        for i in range(num_layers):
            log.info(f"\nConfiguring Layer {i+1}:")
            output_dim = int(typer.prompt(f"  Number of neurons (output size) for Layer {i+1}?"))
            activation_str = typer.prompt(f"  Activation function for Layer {i+1}? (relu, sigmoid, tanh, softmax)", default="relu")
            
            # Add dense layer
            model.add(Dense(current_dim, output_dim))
            
            # Add activation
            if activation_str.lower() == "relu":
                model.add(ActivationLayer(ReLU()))
            elif activation_str.lower() == "sigmoid":
                model.add(ActivationLayer(Sigmoid()))
            elif activation_str.lower() == "tanh":
                model.add(ActivationLayer(Tanh()))
            elif activation_str.lower() == "softmax":
                model.add(ActivationLayer(Softmax()))
            
            current_dim = output_dim
            log.success(f"Added layer: Dense({current_dim}) -> {activation_str}")

        log.info("\nModel architecture summary created.")
        log.success("Your model is ready for training and prediction!")
        
    except ValueError as e:
        log.error("Invalid input. Please enter numbers for layers and dimensions.")
    except Exception as e:
        log.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    app()
